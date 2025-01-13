#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>

#include <iostream>


static int proxy_server;
static sockaddr_in proxy_server_address;

static int proxy_client;
struct sockaddr_in proxy_client_address;

#define DATA_LENGHT 6

static unsigned char* proxy_buffer = new unsigned char[DATA_LENGHT];


void CloseProxy() 
{
    close(proxy_client);
    std::cout << "Proxy client closed" << std::endl;
    close(proxy_server);
    std::cout << "Proxy server closed" << std::endl;
}

bool InitializeProxyServer(int port)
{
    proxy_server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (proxy_server < 0) 
    {
        std::cerr << "Err" << std::endl;
        return false;
    }

    proxy_server_address.sin_family = AF_INET;
    proxy_server_address.sin_addr.s_addr = INADDR_ANY;
    proxy_server_address.sin_port = htons(port);

    if (bind(proxy_server, (sockaddr *)&proxy_server_address, sizeof(proxy_server_address)) < 0) 
    {
        std::cerr << "Err" << std::endl;
        CloseProxy();
        return false;
    }

    if (listen(proxy_server, 1) < 0) 
    {
        std::cerr << "Err" << std::endl;
        CloseProxy();
        return false;
    }

    return true;
}

// bool WaitingForConnection() 
// {
//     socklen_t client_addr_len = sizeof(proxy_client_address);
//     proxy_client = accept(proxy_server, (sockaddr *)&proxy_client_address, &client_addr_len);
//     if (proxy_client < 0) 
//     {
//         std::cerr << "Err" << std::endl;
//         return false;
//     }

//     return true;
// }

static bool proxy_server_connection = false;
unsigned char* GetDataFromProxyServer() 
{
    if (!proxy_server_connection) 
    {
        std::cout << "Waiting for connection" << std::endl;
        socklen_t client_addr_len = sizeof(proxy_client_address);
        proxy_client = accept(proxy_server, (sockaddr *)&proxy_client_address, &client_addr_len);
        proxy_server_connection = true;
        std::cout << "Connected" << std::endl;
    }

    int valread = recv(proxy_client, proxy_buffer, sizeof(proxy_buffer) * DATA_LENGHT, 0);

    if (valread <= 0) 
    {
        proxy_server_connection = false;
        return nullptr;
    }

    return proxy_buffer;
}

