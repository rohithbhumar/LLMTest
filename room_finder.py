from msgraph import GraphServiceClient
from azure.identity.aio import ClientSecretCredential
from msgraph import GraphServiceClient

# credentials = ClientSecretCredential(
#     client_id="ff04a0aa-1bc9-40cd-b5c2-200650b9b22d",
#     client_secret="gGM8Q~FNYTCBCDwPdalmYzCfVv-FwfXpVnWV6csX",
#     tenant_id="bbdc7518-4785-4eca-aa39-ac5a5641716b"
# )
# scopes = ["https://graph.microsoft.com/.default"]
# async def main():
#     graph_client = GraphServiceClient(credentials, scopes)
#     result = await graph_client.users.by_user_id("Rohit Bhumar").get()
#     print(result)
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

# Example using sync credentials and delegated access.
from azure.identity import DeviceCodeCredential
from msgraph import GraphServiceClient

credentials = DeviceCodeCredential(
    client_id="ff04a0aa-1bc9-40cd-b5c2-200650b9b22d",
    tenant_id="bbdc7518-4785-4eca-aa39-ac5a5641716b"
)
scopes = ['User.Read', 'Mail.Read']
client = GraphServiceClient(credentials=credentials, scopes=scopes)


async def main():
    graph_client = GraphServiceClient(credentials, scopes)
    result = await graph_client.me.get()
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
