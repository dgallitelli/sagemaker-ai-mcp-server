"""Helper Functions for SageMaker Managed MLflow."""

from loguru import logger
from sagemaker_ai_mcp_server.helpers.utils import (
    get_sagemaker_client,
    get_sagemaker_execution_role_arn,
)
from typing import Any, Dict, List, Literal


async def list_mlflow_tracking_servers() -> List[Dict[str, Any]]:
    """List all MLflow Tracking Servers in SageMaker.

    Returns:
        List[Dict[str, Any]]: A list of MLflow Tracking Servers.
    """
    client = get_sagemaker_client()
    logger.info('Listing MLflow Tracking Servers...')
    response = client.list_mlflow_tracking_servers()
    return response.get('TrackingServerSummaries', [])


async def create_mlflow_tracking_server(
    tracking_server_name: str,
    artifact_store_uri: str,
    tracking_server_size: Literal['Small', 'Medium', 'Large'],
) -> str:
    """Create an MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server.
        artifact_store_uri (str): The S3 URI for the artifact store.
        tracking_server_size (Literal['Small', 'Medium', 'Large']): The size of the Tracking Server.

    Returns:
        str: The ARN of the created MLflow Tracking Server.
    """
    client = get_sagemaker_client()
    role_arn = get_sagemaker_execution_role_arn()
    logger.info(f'Creating MLflow Tracking Server: {tracking_server_name}')
    response = client.create_mlflow_tracking_server(
        TrackingServerName=tracking_server_name,
        ArtifactStoreUri=artifact_store_uri,
        TrackingServerSize=tracking_server_size,
        RoleArn=role_arn,
    )
    logger.info(f'MLflow Tracking Server {tracking_server_name} created successfully.')
    return response.get('TrackingServerArn', '')


async def create_presigned_mlflow_tracking_server_url(
    tracking_server_name: str,
    expiration_seconds: int = 3600,
) -> str:
    """Create a presigned URL for accessing an MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server.
        expiration_seconds (int): The expiration time for the presigned URL in seconds. Defaults to 3600.

    Returns:
        str: The presigned URL for the MLflow Tracking Server.
    """
    client = get_sagemaker_client()
    logger.info(f'Creating presigned URL for MLflow Tracking Server: {tracking_server_name}')
    response = client.create_presigned_mlflow_tracking_server_url(
        TrackingServerName=tracking_server_name,
        ExpirationSeconds=expiration_seconds,
    )
    return response.get('PresignedUrl', '')


async def describe_mlflow_tracking_server(
    tracking_server_name: str,
) -> Dict[str, Any]:
    """Describe a specific MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server to describe.

    Returns:
        Dict[str, Any]: The details of the specified MLflow Tracking Server.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing MLflow Tracking Server: {tracking_server_name}')
    response = client.describe_mlflow_tracking_server(TrackingServerName=tracking_server_name)
    return response


async def start_mlflow_tracking_server(
    tracking_server_name: str,
) -> Dict[str, Any]:
    """Start a specific MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server to start.

    Returns:
        Dict[str, Any]: The details of the started MLflow Tracking Server.
    """
    client = get_sagemaker_client()
    logger.info(f'Starting MLflow Tracking Server: {tracking_server_name}')
    response = client.start_mlflow_tracking_server(TrackingServerName=tracking_server_name)
    return response


async def stop_mlflow_tracking_server(
    tracking_server_name: str,
) -> Dict[str, Any]:
    """Stop a specific MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server to stop.

    Returns:
        Dict[str, Any]: The details of the stopped MLflow Tracking Server.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping MLflow Tracking Server: {tracking_server_name}')
    response = client.stop_mlflow_tracking_server(TrackingServerName=tracking_server_name)
    return response


async def delete_mlflow_tracking_server(tracking_server_name: str) -> None:
    """Delete an MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting MLflow Tracking Server: {tracking_server_name}')
    client.delete_mlflow_tracking_server(TrackingServerName=tracking_server_name)
    logger.info(f'MLflow Tracking Server {tracking_server_name} deleted successfully.')
