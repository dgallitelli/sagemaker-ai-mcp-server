"""Helper Functions for Profile Ops."""

from loguru import logger
from sagemaker_ai_mcp_server.helpers.utils import get_sagemaker_client
from typing import Any, Dict, List


async def list_user_profiles() -> List[Dict[str, Any]]:
    """List all user profiles in a SageMaker Domain.

    Returns:
        List[Dict[str, Any]]: A list of user profiles in the SageMaker Domain.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker User Profiles...')
    response = client.list_user_profiles()
    return response.get('UserProfiles', [])


async def list_spaces() -> List[Dict[str, Any]]:
    """List all SageMaker Spaces.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Spaces.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Spaces...')
    response = client.list_spaces()
    return response.get('Spaces', [])
