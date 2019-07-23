#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <sys/socket.h>  
#include <netinet/in.h>  
#include <arpa/inet.h>  
#include <netdb.h>  
#include <pthread.h>
#include <unistd.h>

int start_udp_client();
int read_frame_buffer();

int main(int argc, char** argv) {
    read_frame_buffer();
    return 0;
}
