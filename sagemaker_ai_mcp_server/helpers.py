"""The helper functions for the SageMaker AI MCP Server."""

import boto3
import os
from loguru import logger
from typing import Any, Dict, List


def get_region() -> str:
    """Get the AWS region from the environment variable or default to 'us-east-1'.

    Returns:
        str: The AWS region.
    """
    return os.getenv('AWS_REGION', 'us-east-1')


def get_aws_session(region_name=None) -> boto3.Session:
    """Create an AWS session using AWS Profile or default credentials.

    Args:
        region_name (str): The AWS region to use. Defaults to None, which uses the
                           region from the environment variable or defaults to 'us-east-1'.

    Returns:
        boto3.Session: An AWS session object.
    """
    profile_name = os.environ.get('AWS_PROFILE')
    region = region_name or get_region()
    try:
        if profile_name:
            logger.debug(f'Using AWS profile: {profile_name}')
            return boto3.Session(profile_name=profile_name, region_name=region)
        else:
            logger.debug('Using default AWS credential chain')
            return boto3.Session(region_name=region)
    except Exception as e:
        logger.error(f'Error creating AWS session: {e}')
        raise RuntimeError(f'Failed to create AWS session: {e}')


def get_sagemaker_client(region_name=None):
    """Get a SageMaker client.

    Args:
        region_name (str): The AWS region to use. Defaults to None, which uses the
                           region from the environment variable or defaults to 'us-east-1'.

    Returns:
        boto3.client: A SageMaker client object.
    """
    session = get_aws_session(region_name)
    return session.client('sagemaker')


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


async def describe_training_job(training_job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Training Job."""
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Training Job: {training_job_name}')
    response = client.describe_training_job(TrainingJobName=training_job_name)
    return response


async def list_training_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Training Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Training Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Training Jobs...')
    response = client.list_training_jobs()
    return response.get('TrainingJobSummaries', [])


async def stop_training_job(training_job_name: str) -> None:
    """Stop a SageMaker Training Job.

    Args:
        training_job_name (str): The name of the SageMaker Training Job to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Training Job: {training_job_name}')
    client.stop_training_job(TrainingJobName=training_job_name)
    logger.info(f'Training Job {training_job_name} stopped successfully.')


async def describe_processing_job(processing_job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Processing Job.

    Args:
        processing_job_name (str): The name of the SageMaker Processing Job to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Processing Job.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Processing Job: {processing_job_name}')
    response = client.describe_processing_job(ProcessingJobName=processing_job_name)
    return response


async def list_processing_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Processing Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Processing Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Processing Jobs...')
    response = client.list_processing_jobs()
    return response.get('ProcessingJobSummaries', [])


async def stop_processing_job(processing_job_name: str) -> None:
    """Stop a SageMaker Processing Job.

    Args:
        processing_job_name (str): The name of the SageMaker Processing Job to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Processing Job: {processing_job_name}')
    client.stop_processing_job(ProcessingJobName=processing_job_name)
    logger.info(f'Processing Job {processing_job_name} stopped successfully.')


async def list_transform_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Transform Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Transform Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Transform Jobs...')
    response = client.list_transform_jobs()
    return response.get('TransformJobSummaries', [])


async def describe_transform_job(transform_job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Transform Job.

    Args:
        transform_job_name (str): The name of the SageMaker Transform Job to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Transform Job.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Transform Job: {transform_job_name}')
    response = client.describe_transform_job(TransformJobName=transform_job_name)
    return response


async def stop_transform_job(transform_job_name: str) -> None:
    """Stop a SageMaker Transform Job.

    Args:
        transform_job_name (str): The name of the SageMaker Transform Job to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Transform Job: {transform_job_name}')
    client.stop_transform_job(TransformJobName=transform_job_name)
