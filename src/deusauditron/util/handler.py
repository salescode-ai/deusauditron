import asyncio
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from deusauditron.app_logging.logger import logger
from deusauditron.config import get_config
import aiohttp
from pydantic import ValidationError
from deusauditron.schemas.shared_models.models import User

async def get_authorization(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
):
    if credentials is None:
        logger.warning("Authentication failed: No token provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A bearer token is required for this endpoint.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    auth_header = f"Bearer {credentials.credentials.removeprefix('Bearer ')}"
    logger.debug("Attempting token validation with authentication service")
    
    user_data = {}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(
                get_config().mgmt_url + "/users/me",
                headers={"Authorization": auth_header}
            ) as response:
                response.raise_for_status()
                user_data = await response.json()
                
        user = User(**user_data)

        if not user.enabled:
            logger.warning(
                "Authentication failed for disabled user",
                extra={"username": user.userName, "user_id": user.userId}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled.",
            )

        logger.info(
            "Successfully authenticated user",
            extra={"username": user.userName, "user_id": user.userId}
        )
        return user
        
    except asyncio.TimeoutError as e:
        logger.error(f"Authentication service timeout: {e}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Authentication service is currently unavailable.",
        )
    except aiohttp.ClientResponseError as e:
        logger.error(f"Authentication service returned error: {e.status} - {e.message}")
        if e.status in {401, 403}:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service is currently unavailable.",
            )
    except ValidationError as e:
        logger.error(f"Invalid response from authentication service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Received an invalid response from the authentication service.",
        )
    except Exception as e:
        logger.error(f"Authentication service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during authentication.",
        )