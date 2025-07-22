"""The main file for the SageMaker AI MCP Server."""

from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from sagemaker_ai_mcp_server.helpers import (
    delete_endpoint,
    delete_endpoint_config,
    describe_endpoint,
    describe_endpoint_config,
    list_endpoint_configs,
    list_endpoints,
)
from typing import Annotated, Any, Dict, List


mcp = FastMCP(
    'sagemaker-ai-mcp-server',
    instructions="""
    SageMaker AI MCP Server provides tools to interact with Amazon SageMaker AI
    service.
    
    The server enables you to:
    - List SageMaker Endpoints
    - List SageMaker Endpoint Configurations
    - Delete SageMaker Endpoints
    - Delete SageMaker Endpoint Configurations
    - Describe SageMaker Endpoints
    - Describe SageMaker Endpoint Configurations
    
    Use these tools to manage your SageMaker resources effectively.
    """,
    dependencies=[
        'pydantic',
        'loguru',
        'boto3',
    ],
)


@mcp.tool(name='list_endpoints_sagemaker', description='List all SageMaker Endpoints')
async def list_endpoints_sagemaker() -> Dict[str, List]:
    """Get a list of all SageMaker Endpoint.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Endpoints in your account
    in the current region. This is typically used first to see what endpoints
    are available before performing operations on them.

    ## Example

    ```python
    endpoints = await list_endpoints_sagemaker()
    print(endpoints)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'endpoints': A list of dictionaries, each representing a SageMaker
      Endpoint with its details.

    ## Returns
    A dictionary containing a list of SageMaker Endpoints.
    """
    try:
        endpoints = await list_endpoints()
        return {'endpoints': endpoints}
    except Exception as e:
        logger.error(f'Error listing SageMaker endpoints: {e}')
        raise ValueError(f'Failed to list endpoints: {e}')


@mcp.tool(
    name='list_endpoint_configs_sagemaker',
    description='List all SageMaker Endpoint Configurations',
)
async def list_endpoint_configs_sagemaker() -> Dict[str, List]:
    """Get a list of all SageMaker Endpoint Configurations.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Endpoint Configurations
    in your account in the current region. This helps you identify available
    endpoint configurations before performing operations on them.

    ## Example

    ```python
    configs = await list_endpoint_configs_sagemaker()
    print(configs)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'endpoint_configs': A list of dictionaries, each representing a SageMaker
      Endpoint Configuration with its details.

    ## Returns
    A dictionary containing a list of SageMaker Endpoint Configurations.
    """
    try:
        configs = await list_endpoint_configs()
        return {'endpoint_configs': configs}
    except Exception as e:
        logger.error(f'Error listing endpoint configurations: {e}')
        raise ValueError(f'Failed to list endpoint configs: {e}')


@mcp.tool(name='delete_endpoint_sagemaker', description='Delete a SageMaker Endpoint')
async def delete_endpoint_sagemaker(
    endpoint_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint to delete')
    ],
) -> Dict[str, str]:
    """Delete a specified SageMaker Endpoint.

    ## Usage

    Use this tool to delete a SageMaker Endpoint by providing its name.
    This is useful for cleaning up resources that are no longer needed.

    ## Example

    ```python
    result = await delete_endpoint_sagemaker(endpoint_name='my-endpoint')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_endpoint(endpoint_name)
        return {'message': f"Endpoint '{endpoint_name}' deleted successfully"}
    except Exception as e:
        logger.error(f'Error deleting endpoint {endpoint_name}: {e}')
        raise ValueError(f'Failed to delete endpoint {endpoint_name}: {e}')


@mcp.tool(
    name='delete_endpoint_config_sagemaker',
    description='Delete a SageMaker Endpoint Configuration',
)
async def delete_endpoint_config_sagemaker(
    endpoint_config_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint Configuration to delete')
    ],
) -> Dict[str, str]:
    """Delete a specified SageMaker Endpoint Configuration.

    ## Usage

    Use this tool to delete a SageMaker Endpoint Configuration by providing
    its name. This is useful for cleaning up configurations that are no longer
    needed.

    ## Example

    ```python
    result = await delete_endpoint_config_sagemaker(endpoint_config_name='my-endpoint-config')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_endpoint_config(endpoint_config_name)
        msg = f"Endpoint Config '{endpoint_config_name}' deleted successfully"
        return {'message': msg}
    except Exception as e:
        logger.error(f'Error deleting config {endpoint_config_name}: {e}')
        raise ValueError(f'Failed to delete config {endpoint_config_name}: {e}')


@mcp.tool(name='describe_endpoint_sagemaker', description='Describe a SageMaker Endpoint')
async def describe_endpoint_sagemaker(
    endpoint_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Endpoint.

    ## Usage

    Use this tool to get detailed information about a SageMaker Endpoint by
    providing its name. This returns comprehensive information about the
    endpoint's configuration, status, and other details.

    ## Example

    ```python
    details = await describe_endpoint_sagemaker(endpoint_name='my-endpoint')
    print(details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Endpoint.

    ## Returns
    A dictionary containing the endpoint details.
    """
    try:
        endpoint_details = await describe_endpoint(endpoint_name)
        return endpoint_details
    except Exception as e:
        logger.error(f'Error describing endpoint {endpoint_name}: {e}')
        raise ValueError(f'Failed to describe endpoint {endpoint_name}: {e}')


@mcp.tool(
    name='describe_endpoint_config_sagemaker',
    description='Describe a SageMaker Endpoint Configuration',
)
async def describe_endpoint_config_sagemaker(
    endpoint_config_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint Configuration to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Endpoint Configuration.

    ## Usage

    Use this tool to get detailed information about a SageMaker Endpoint
    Configuration by providing its name. This returns comprehensive information
    about the configuration details, including model specifications, instance
    types, and other configuration parameters.

    ## Example

    ```python
    config_details = await describe_endpoint_config_sagemaker(
        endpoint_config_name='my-endpoint-config'
    )
    print(config_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Endpoint Configuration.

    ## Returns
    A dictionary containing the endpoint configuration details.
    """
    try:
        config_details = await describe_endpoint_config(endpoint_config_name)
        return config_details
    except Exception as e:
        logger.error(f'Error describing config {endpoint_config_name}: {e}')
        err_msg = f'Failed to describe config {endpoint_config_name}: {e}'
        raise ValueError(err_msg)


def main():
    """Run the SageMaker AI MCP Server."""
    logger.info('Starting SageMaker AI MCP Server...')
    mcp.run()


if __name__ == '__main__':
    main()
