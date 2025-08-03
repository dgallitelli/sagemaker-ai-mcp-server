"""Utils Functions for Region, Execution Role, Sessions, SageMaker client."""

import boto3
import os
from loguru import logger


def get_region() -> str:
    """Get the AWS region from the environment variable or default to 'us-east-1'.

    Returns:
        str: The AWS region.
    """
    return os.getenv('AWS_REGION', 'us-east-1')


def get_sagemaker_execution_role_arn() -> str:
    """Get the SageMaker execution role ARN from the environment variable.

    Returns:
        str: The SageMaker execution role ARN.
    """
    role_arn = os.environ.get('SAGEMAKER_EXECUTION_ROLE_ARN')
    if not role_arn:
        raise ValueError('SAGEMAKER_EXECUTION_ROLE_ARN environment variable is not set.')
    return role_arn


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
