"""Helper Functions for SageMaker Models"""

from loguru import logger
from sagemaker_ai_mcp_server.helpers.utils import get_sagemaker_client
from typing import Any, Dict, List


async def list_models() -> List[Dict[str, Any]]:
    """List all SageMaker Models.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Models.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Models...')
    response = client.list_models()
    return response.get('Models', [])


async def describe_model(model_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Model.

    Args:
        model_name (str): The name of the SageMaker Model to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Model.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Model: {model_name}')
    response = client.describe_model(ModelName=model_name)
    return response


async def delete_model(model_name: str) -> None:
    """Delete a SageMaker Model.

    Args:
        model_name (str): The name of the SageMaker Model to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Model: {model_name}')
    client.delete_model(ModelName=model_name)
    logger.info(f'Model {model_name} deleted successfully.')
