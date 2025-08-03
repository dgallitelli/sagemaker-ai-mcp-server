"""Helper Functions for SageMaker Domains"""

from loguru import logger
from sagemaker_ai_mcp_server.helpers.utils import get_sagemaker_client
from typing import Any, Dict, List


async def list_domains() -> List[Dict[str, Any]]:
    """List all SageMaker Domains.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Domains.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Domains...')
    response = client.list_domains()
    return response.get('Domains', [])


async def create_presigned_domain_url(
    domain_id: str,
    user_profile_name: str,
    expiration_seconds: int = 3600,
) -> str:
    """Create a presigned URL for accessing a SageMaker Domain.

    Args:
        domain_id (str): The ID of the SageMaker Domain.
        user_profile_name (str): The name of the user profile.
        expiration_seconds (int): The expiration time for the presigned URL in seconds. Defaults to 3600.

    Returns:
        str: The presigned URL for the SageMaker Domain.
    """
    client = get_sagemaker_client()
    logger.info(f'Creating presigned URL for SageMaker Domain: {domain_id}')
    response = client.create_presigned_domain_url(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        ExpirationSeconds=expiration_seconds,
    )
    return response.get('AuthorizedUrl', '')


async def describe_domain(domain_id: str) -> Dict[str, Any]:
    """Describe a specific SageMaker Domain.

    Args:
        domain_id (str): The ID of the SageMaker Domain to describe.

    Returns:
        Dict[str, Any]: The details of the specified SageMaker Domain.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Domain: {domain_id}')
    response = client.describe_domain(DomainId=domain_id)
    return response


async def delete_domain(domain_id: str) -> None:
    """Delete a SageMaker Domain.

    Args:
        domain_id (str): The ID of the SageMaker Domain to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Domain: {domain_id}')
    client.delete_domain(DomainId=domain_id)
    logger.info(f'Domain {domain_id} deleted successfully.')
