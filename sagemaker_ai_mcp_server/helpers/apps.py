"""Helper Functions for SageMaker Apps."""

from loguru import logger
from sagemaker_ai_mcp_server.helpers.utils import get_sagemaker_client
from typing import Any, Dict, List, Literal


async def list_apps() -> List[Dict[str, Any]]:
    """List all SageMaker Apps.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Apps.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Apps...')

    response = client.list_apps()
    return response.get('Apps', [])


async def create_app(
    domain_id: str,
    user_profile_name: str,
    app_type: Literal[
        'JupyterServer',
        'KernelGateway',
        'RStudioServerPro',
        'RSessionGateway',
        'Canvas',
        'JupyterLab',
        'CodeEditor',
        'TensorBoard',
        'DetailedProfiler',
    ],
    app_name: str,
    resource_spec: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Create a SageMaker App.

    Args:
        domain_id (str): The ID of the domain in which to create the app.
        user_profile_name (str): The name of the user profile in which to create the app.
        app_type (str): The type of app to create, e.g., 'JupyterServer', 'KernelGateway'.
        app_name (str): The name of the app.
        resource_spec (Dict[str, Any], optional): The resource specification for the app. Defaults to None.

    Returns:
        Dict[str, Any]: The details of the created SageMaker App.
    """
    client = get_sagemaker_client()
    logger.info(
        f'Creating SageMaker App: {app_name} of type {app_type} for user {user_profile_name} in domain {domain_id}'
    )

    create_params = {
        'DomainId': domain_id,
        'UserProfileName': user_profile_name,
        'AppType': app_type,
        'AppName': app_name,
    }

    if resource_spec:
        create_params['ResourceSpec'] = resource_spec

    response = client.create_app(**create_params)
    logger.info(f'App {app_name} creation initiated successfully.')
    return response.get('AppArn', {})


async def create_presigned_notebook_instance_url(
    notebook_instance_name: str,
    session_expiration_duration_in_seconds: int = 3600,
) -> str:
    """Create a presigned URL for accessing a SageMaker Notebook Instance.

    Args:
        notebook_instance_name (str): The name of the SageMaker Notebook Instance.
        session_expiration_duration_in_seconds (int, optional): The expiration time for the presigned URL in seconds. Defaults to 3600.

    Returns:
        str: The presigned URL for the SageMaker Notebook Instance.
    """
    client = get_sagemaker_client()
    logger.info(
        f'Creating presigned URL for SageMaker Notebook Instance: {notebook_instance_name}'
    )
    response = client.create_presigned_notebook_instance_url(
        NotebookInstanceName=notebook_instance_name,
        SessionExpirationDurationInSeconds=session_expiration_duration_in_seconds,
    )
    return response.get('AuthorizedUrl', '')


async def describe_app(
    domain_id: str,
    user_profile_name: str,
    app_type: Literal[
        'JupyterServer',
        'KernelGateway',
        'RStudioServerPro',
        'RSessionGateway',
        'Canvas',
        'JupyterLab',
        'CodeEditor',
        'TensorBoard',
        'DetailedProfiler',
    ],
    app_name: str,
) -> Dict[str, Any]:
    """Describe a SageMaker App.

    Args:
        domain_id (str): The ID of the domain in which the app resides.
        user_profile_name (str): The name of the user profile that owns the app.
        app_type (str): The type of app, e.g., 'JupyterServer', 'KernelGateway'.
        app_name (str): The name of the app.

    Returns:
        Dict[str, Any]: The details of the SageMaker App.
    """
    client = get_sagemaker_client()
    logger.info(
        f'Describing SageMaker App: {app_name} of type {app_type} for user {user_profile_name} in domain {domain_id}'
    )
    response = client.describe_app(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        AppType=app_type,
        AppName=app_name,
    )
    return response


async def describe_app_image_config(app_image_config_name: str) -> Dict[str, Any]:
    """Describe a SageMaker App Image Config.

    Args:
        app_image_config_name (str): The name of the SageMaker App Image Config to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker App Image Config.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker App Image Config: {app_image_config_name}')
    response = client.describe_app_image_config(AppImageConfigName=app_image_config_name)
    return response


async def delete_app(
    domain_id: str,
    user_profile_name: str,
    app_type: Literal[
        'JupyterServer',
        'KernelGateway',
        'RStudioServerPro',
        'RSessionGateway',
        'Canvas',
        'JupyterLab',
        'CodeEditor',
        'TensorBoard',
        'DetailedProfiler',
    ],
    app_name: str,
) -> None:
    """Delete a SageMaker App.

    Args:
        domain_id (str): The ID of the domain in which the app resides.
        user_profile_name (str): The name of the user profile that owns the app.
        app_type (str): The type of app to delete, e.g., 'JupyterServer', 'KernelGateway'.
        app_name (str): The name of the app to delete.
    """
    client = get_sagemaker_client()
    logger.info(
        f'Deleting SageMaker App: {app_name} of type {app_type} for user {user_profile_name} in domain {domain_id}'
    )
    client.delete_app(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        AppType=app_type,
        AppName=app_name,
    )
    logger.info(f'App {app_name} deletion initiated successfully.')


async def delete_app_image_config(app_image_config_name: str) -> None:
    """Delete a SageMaker App Image Config.

    Args:
        app_image_config_name (str): The name of the SageMaker App Image Config to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker App Image Config: {app_image_config_name}')
    client.delete_app_image_config(AppImageConfigName=app_image_config_name)
    logger.info(f'App Image Config {app_image_config_name} deleted successfully.')
