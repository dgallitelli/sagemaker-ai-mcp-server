"""Tests for SageMaker AI MLFlow Managed Tracking Servers."""

import pytest
from sagemaker_ai_mcp_server.helpers.mlflow_managed import (
    create_mlflow_tracking_server,
    create_presigned_mlflow_tracking_server_url,
    delete_mlflow_tracking_server,
    describe_mlflow_tracking_server,
    list_mlflow_tracking_servers,
    start_mlflow_tracking_server,
    stop_mlflow_tracking_server,
)
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.mlflow_managed.get_sagemaker_client')
async def test_list_mlflow_tracking_servers(mock_get_sagemaker_client):
    """Test listing SageMaker AI MLFlow Tracking Servers."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    mock_response = {
        'TrackingServerSummaries': [
            {'TrackingServerName': 'test-mlflow-server', 'Status': 'InService'}
        ]
    }
    mock_client.list_mlflow_tracking_servers.return_value = mock_response
    servers = await list_mlflow_tracking_servers()
    mock_get_sagemaker_client.assert_called_once()
    mock_client.list_mlflow_tracking_servers.assert_called_once()
    expected = [{'TrackingServerName': 'test-mlflow-server', 'Status': 'InService'}]
    assert servers == expected


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.mlflow_managed.get_sagemaker_client')
@patch('sagemaker_ai_mcp_server.helpers.mlflow_managed.get_sagemaker_execution_role_arn')
async def test_create_mlflow_tracking_server(mock_get_role_arn, mock_get_sagemaker_client):
    """Test creating a SageMaker AI MLFlow Tracking Server."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    role_arn = 'arn:aws:iam::123456789012:role/AmazonSageMaker-ExecutionRole'
    mock_get_role_arn.return_value = role_arn
    await create_mlflow_tracking_server('test-mlflow-server', 's3://bucket/artifacts', 'Medium')
    mock_get_sagemaker_client.assert_called_once()
    mock_get_role_arn.assert_called_once()
    mock_client.create_mlflow_tracking_server.assert_called_once_with(
        TrackingServerName='test-mlflow-server',
        ArtifactStoreUri='s3://bucket/artifacts',
        TrackingServerSize='Medium',
        RoleArn=role_arn,
    )


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.mlflow_managed.get_sagemaker_client')
async def test_create_presigned_mlflow_tracking_server_url_default(mock_get_sagemaker_client):
    """Test creating a presigned URL for a SageMaker AI MLFlow Tracking Server."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    expected_response = {'PresignedUrl': 'https://example.com/presigned-url'}
    mock_client.create_presigned_mlflow_tracking_server_url.return_value = expected_response
    url = await create_presigned_mlflow_tracking_server_url('test-mlflow-server')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.create_presigned_mlflow_tracking_server_url.assert_called_once_with(
        TrackingServerName='test-mlflow-server', ExpirationSeconds=3600
    )
    assert url == 'https://example.com/presigned-url'


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.mlflow_managed.get_sagemaker_client')
async def test_create_presigned_mlflow_tracking_server_url_custom(mock_get_sagemaker_client):
    """Test creating a presigned URL for a SageMaker AI MLFlow Tracking Server."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    expected_response = {'PresignedUrl': 'https://example.com/presigned-url-custom'}
    mock_client.create_presigned_mlflow_tracking_server_url.return_value = expected_response
    custom_expiration = 7200
    url = await create_presigned_mlflow_tracking_server_url(
        'test-mlflow-server', custom_expiration
    )
    mock_get_sagemaker_client.assert_called_once()
    mock_client.create_presigned_mlflow_tracking_server_url.assert_called_once_with(
        TrackingServerName='test-mlflow-server', ExpirationSeconds=custom_expiration
    )
    assert url == 'https://example.com/presigned-url-custom'


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.mlflow_managed.get_sagemaker_client')
async def test_describe_mlflow_tracking_server(mock_get_sagemaker_client):
    """Test describing a SageMaker AI MLFlow Tracking Server."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    expected_response = {
        'TrackingServerName': 'test-mlflow-server',
        'Status': 'InService',
        'CreationTime': '2023-01-01T00:00:00Z',
    }
    mock_client.describe_mlflow_tracking_server.return_value = expected_response
    response = await describe_mlflow_tracking_server('test-mlflow-server')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.describe_mlflow_tracking_server.assert_called_once_with(
        TrackingServerName='test-mlflow-server'
    )
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.mlflow_managed.get_sagemaker_client')
async def test_start_mlflow_tracking_server(mock_get_sagemaker_client):
    """Test starting a SageMaker AI MLFlow Tracking Server."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    expected_response = {'TrackingServerName': 'test-mlflow-server', 'Status': 'Starting'}
    mock_client.start_mlflow_tracking_server.return_value = expected_response
    response = await start_mlflow_tracking_server('test-mlflow-server')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.start_mlflow_tracking_server.assert_called_once_with(
        TrackingServerName='test-mlflow-server'
    )
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.mlflow_managed.get_sagemaker_client')
async def test_stop_mlflow_tracking_server(mock_get_sagemaker_client):
    """Test stopping a SageMaker AI MLFlow Tracking Server."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    expected_response = {'TrackingServerName': 'test-mlflow-server', 'Status': 'Stopping'}
    mock_client.stop_mlflow_tracking_server.return_value = expected_response
    response = await stop_mlflow_tracking_server('test-mlflow-server')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.stop_mlflow_tracking_server.assert_called_once_with(
        TrackingServerName='test-mlflow-server'
    )
    assert response == expected_response


@pytest.mark.asyncio
@patch('sagemaker_ai_mcp_server.helpers.mlflow_managed.get_sagemaker_client')
async def test_delete_mlflow_tracking_server(mock_get_sagemaker_client):
    """Test deleting a SageMaker AI MLFlow Tracking Server."""
    mock_client = MagicMock()
    mock_get_sagemaker_client.return_value = mock_client
    await delete_mlflow_tracking_server('test-mlflow-server')
    mock_get_sagemaker_client.assert_called_once()
    mock_client.delete_mlflow_tracking_server.assert_called_once_with(
        TrackingServerName='test-mlflow-server'
    )
