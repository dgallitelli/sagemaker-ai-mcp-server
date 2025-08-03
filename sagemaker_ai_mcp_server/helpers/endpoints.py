"""Helper Functions for SageMaker Endpoints."""

from loguru import logger
from sagemaker_ai_mcp_server.helpers.utils import get_sagemaker_client
from typing import Any, Dict, List


async def list_endpoints() -> List[Dict[str, Any]]:
    """List all SageMaker Endpoints that are available.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Endpoints.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Endpoints...')
    response = client.list_endpoints()
    return response.get('Endpoints', [])


async def list_endpoint_configs() -> List[Dict[str, Any]]:
    """List all SageMaker Endpoint Configurations.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Endpoint Configurations.
    """
    client = get_sagemaker_client()
    response = client.list_endpoint_configs()
    logger.info('Listing SageMaker Endpoint Configurations...')
    return response.get('EndpointConfigs', [])


async def describe_endpoint(endpoint_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Endpoint.

    Args:
        endpoint_name (str): The name of the SageMaker Endpoint to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Endpoint.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Endpoint: {endpoint_name}')
    response = client.describe_endpoint(EndpointName=endpoint_name)
    return response


async def describe_endpoint_config(endpoint_config_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Endpoint Configuration.

    Args:
        endpoint_config_name (str): The name of the SageMaker Endpoint Configuration to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Endpoint Configuration.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Endpoint Config: {endpoint_config_name}')
    response = client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
    return response


async def delete_endpoint(endpoint_name: str) -> None:
    """Delete a SageMaker Endpoint.

    Args:
        endpoint_name (str): The name of the SageMaker Endpoint to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Endpoint: {endpoint_name}')
    client.delete_endpoint(EndpointName=endpoint_name)
    logger.info(f'Endpoint {endpoint_name} deleted successfully.')


async def delete_endpoint_config(endpoint_config_name: str) -> None:
    """Delete a SageMaker Endpoint Configuration.

    Args:
        endpoint_config_name (str): The name of the SageMaker Endpoint Configuration to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Endpoint Config: {endpoint_config_name}')
    client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    logger.info(f'Endpoint Config {endpoint_config_name} deleted successfully.')
