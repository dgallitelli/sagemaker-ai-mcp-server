"""Tests for SageMaker AI Apps."""

import pytest
from sagemaker_ai_mcp_server.helpers.apps import (
    create_app,
    create_presigned_notebook_instance_url,
    delete_app,
    delete_app_image_config,
    describe_app,
    describe_app_image_config,
    list_apps,
)
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.apps.get_sagemaker_client')
async def test_list_apps(mock_get_sagemaker_client):
    """Test listing SageMaker AI Apps."""
    mock_client = MagicMock()
    mock_client.list_apps.return_value = {
        'Apps': [
            {
                'DomainId': 'test-domain',
                'UserProfileName': 'test-user',
                'AppType': 'JupyterServer',
                'AppName': 'test-app-1',
            },
            {
                'DomainId': 'test-domain',
                'UserProfileName': 'test-user',
                'AppType': 'KernelGateway',
                'AppName': 'test-app-2',
            },
        ]
    }
    mock_get_sagemaker_client.return_value = mock_client
    apps = await list_apps()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_apps.assert_called_once()
    assert len(apps) == 2
    assert apps[0]['AppName'] == 'test-app-1'
    assert apps[1]['AppName'] == 'test-app-2'


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.apps.get_sagemaker_client')
async def test_create_app(mock_get_sagemaker_client):
    """Test creating a SageMaker AI App."""
    mock_client = MagicMock()
    mock_client.create_app.return_value = {
        'AppArn': 'arn:aws:sagemaker:us-west-2:123456789012:app/domain-id/user/app-name'
    }
    mock_get_sagemaker_client.return_value = mock_client
    domain_id = 'test-domain'
    user_profile_name = 'test-user'
    app_type = 'JupyterServer'
    app_name = 'test-app'
    resource_spec = {'InstanceType': 'ml.t3.medium'}
    app_arn = await create_app(domain_id, user_profile_name, app_type, app_name, resource_spec)
    mock_get_sagemaker_client.assert_called_once()
    mock_client.create_app.assert_called_once_with(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        AppType=app_type,
        AppName=app_name,
        ResourceSpec=resource_spec,
    )
    assert app_arn == 'arn:aws:sagemaker:us-west-2:123456789012:app/domain-id/user/app-name'


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.apps.get_sagemaker_client')
async def test_create_presigned_notebook_instance_url(mock_get_sagemaker_client):
    """Test creating a presigned notebook instance URL."""
    mock_client = MagicMock()
    mock_client.create_presigned_notebook_instance_url.return_value = {
        'AuthorizedUrl': 'https://example.com/presigned-notebook-url'
    }
    mock_get_sagemaker_client.return_value = mock_client
    notebook_name = 'test-notebook'
    expiration = 7200
    url = await create_presigned_notebook_instance_url(notebook_name, expiration)
    mock_get_sagemaker_client.assert_called_once()
    mock_client.create_presigned_notebook_instance_url.assert_called_once_with(
        NotebookInstanceName=notebook_name,
        SessionExpirationDurationInSeconds=expiration,
    )
    assert url == 'https://example.com/presigned-notebook-url'


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.apps.get_sagemaker_client')
async def test_describe_app(mock_get_sagemaker_client):
    """Test describing a SageMaker AI App."""
    mock_client = MagicMock()
    expected_response = {
        'DomainId': 'test-domain',
        'UserProfileName': 'test-user',
        'AppType': 'JupyterServer',
        'AppName': 'test-app',
        'Status': 'InService',
    }
    mock_client.describe_app.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    domain_id = 'test-domain'
    user_profile_name = 'test-user'
    app_type = 'JupyterServer'
    app_name = 'test-app'
    response = await describe_app(domain_id, user_profile_name, app_type, app_name)
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_app.assert_called_once_with(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        AppType=app_type,
        AppName=app_name,
    )
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.apps.get_sagemaker_client')
async def test_describe_app_image_config(mock_get_sagemaker_client):
    """Test describing a SageMaker AI App Image Config."""
    mock_client = MagicMock()
    expected_response = {
        'AppImageConfigName': 'test-config',
        'CreationTime': '2023-01-01T00:00:00Z',
    }
    mock_client.describe_app_image_config.return_value = expected_response
    mock_get_sagemaker_client.return_value = mock_client
    config_name = 'test-config'
    response = await describe_app_image_config(config_name)
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_app_image_config.assert_called_once_with(AppImageConfigName=config_name)
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.apps.get_sagemaker_client')
async def test_delete_app(mock_get_sagemaker_client):
    """Test deleting a SageMaker AI App."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    domain_id = 'test-domain'
    user_profile_name = 'test-user'
    app_type = 'JupyterServer'
    app_name = 'test-app'
    await delete_app(domain_id, user_profile_name, app_type, app_name)
    mock_get_sagemaker_client.assert_called_once()
    mock_client.delete_app.assert_called_once_with(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        AppType=app_type,
        AppName=app_name,
    )


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.apps.get_sagemaker_client')
async def test_delete_app_image_config(mock_get_sagemaker_client):
    """Test deleting a SageMaker AI App Image Config."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    config_name = 'test-app-image-config'
    await delete_app_image_config(config_name)
    mock_get_sagemaker_client.assert_called_once()
    mock_client.delete_app_image_config.assert_called_once_with(AppImageConfigName=config_name)
