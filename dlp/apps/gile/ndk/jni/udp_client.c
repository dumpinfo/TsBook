#include <stdio.h>  
#include <stdlib.h>  
  #include <string.h>  
#include <sys/socket.h>  
#include <netinet/in.h>  
#include <arpa/inet.h>  
#include <netdb.h>  
#include <pthread.h>
#include <unistd.h>

void run_send_thread(char* send_params);
void run_recv_thread(char* recv_params);
  
int port=8088;  
int start_udp_client() {
    pthread_t send_thread;
    pthread_t recv_thread;
    char send_params[1024];
    char recv_params[1024];
    
    sprintf(send_params, "{\"name\": \"yt\"}");
    pthread_create(&send_thread, NULL, (void*)run_send_thread, (void*)send_params);
    sprintf(recv_params, "{\"method\":\"receive\"}");
    pthread_create(&recv_thread, NULL, (void*)run_recv_thread, (void*)recv_params);
    pthread_join(send_thread, NULL);
    pthread_join(recv_thread, NULL);
    return 0;
}

void run_send_thread(char* send_params)
{
    int socket_descriptor; //套接口描述字  
    int iter=0;  
    char buf[80];  
    //char read_buf[1024];
    //char msg[1024];
    struct sockaddr_in address;//处理网络通信的地址
    
    printf("send_params:%s\r\n", send_params);
  
    bzero(&address,sizeof(address));  
    address.sin_family=AF_INET;  
    address.sin_addr.s_addr=inet_addr("47.92.53.13");//这里不一样  
    address.sin_port=htons(port);
    //创建一个 UDP socket  
    socket_descriptor=socket(AF_INET,SOCK_DGRAM,0);//IPV4  SOCK_DGRAM 数据报套接字（UDP协议）  
    socklen_t len = sizeof(address);
    buf[0] = 1;
    sendto(socket_descriptor, buf, 1, 0, (struct sockaddr *)&address, sizeof(address));
    for(iter=0;iter<=20;iter++)  
    {  
         /* 
         * sprintf(s, "%8d%8d", 123, 4567); //产生：" 123 4567"  
         * 将格式化后到 字符串存放到s当中 
         */  
        sprintf(buf,"dp:%d\n",iter);
        printf("*****loop:%d\r\n", iter);
        sendto(socket_descriptor, buf, strlen(buf),0,(struct sockaddr *)&address,sizeof(address));  
        /*recvfrom(socket_descriptor, read_buf, sizeof(read_buf), 0, (struct sockaddr *)&address, &len);
        printf("         step 2\r\n");
        sprintf(msg, "received:%s\r\n", read_buf);
        buf[0] = '\0';
        read_buf[0] = '\0';
        printf("          step 3\r\n");
        printf("%s", msg);
        msg[0] = '\0';*/
    }
  
    sprintf(buf,"stop\n");  
    sendto(socket_descriptor,buf, strlen(buf),0,(struct sockaddr *)&address,sizeof(address));//发送stop 命令  
    close(socket_descriptor);  
    printf("Messages Sent,terminating\n");
}

void run_recv_thread(char* recv_params)
{
    int socket_descriptor; //套接口描述字  
    int iter=0;  
    //char buf[80];  
    char read_buf[1024];
    char msg[1024];
    struct sockaddr_in address;//处理网络通信的地址
    
    printf("recv_params:%s\r\n", recv_params);
  
    bzero(&address,sizeof(address));  
    address.sin_family=AF_INET;  
    address.sin_addr.s_addr=inet_addr("47.92.53.13");//这里不一样  
    address.sin_port=htons(port);
    //创建一个 UDP socket  
    socket_descriptor=socket(AF_INET,SOCK_DGRAM,0);//IPV4  SOCK_DGRAM 数据报套接字（UDP协议）  
    socklen_t len = sizeof(address);
    msg[0] = 2;
    sendto(socket_descriptor, msg, 1, 0, (struct sockaddr *)&address, sizeof(address));
    for(iter=0;iter<=20;iter++)  
    {  
         /* 
         * sprintf(s, "%8d%8d", 123, 4567); //产生：" 123 4567"  
         * 将格式化后到 字符串存放到s当中 
         */
        // sendto(socket_descriptor, buf, strlen(buf),0,(struct sockaddr *)&address,sizeof(address));  
        recvfrom(socket_descriptor, read_buf, sizeof(read_buf), 0, (struct sockaddr *)&address, &len);
        printf("         step 2\r\n");
        sprintf(msg, "received:%s\r\n", read_buf);
        read_buf[0] = '\0';
        printf("          step 3\r\n");
        printf("%s", msg);
        msg[0] = '\0';
    }
  
    sprintf(msg, "recv stop\n");  
    sendto(socket_descriptor,msg, strlen(msg),0,(struct sockaddr *)&address,sizeof(address));//发送stop 命令  
    close(socket_descriptor);  
    printf("Messages received,terminating\n");
}

