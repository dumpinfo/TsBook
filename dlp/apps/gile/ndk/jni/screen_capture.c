#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <sys/socket.h>  
#include <netinet/in.h>  
#include <arpa/inet.h>  
#include <netdb.h>  
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/fb.h>
#include <sys/mman.h>
#include "png/png.h"

int get_fb_fix_screeninfo(int fd, struct fb_fix_screeninfo** finfo);
int get_fb_var_screeninfo(int fd, struct fb_var_screeninfo** vinfo);
int get_frame_buffer_rgb();
void read_png_file(char *filename);
void write_png_file_data(char *filename, png_bytep *pp_row_pointers, int img_width, int img_height);

int read_frame_buffer()
{
    int fd;
    int rst;
    struct fb_fix_screeninfo* finfo = NULL;
    struct fb_var_screeninfo* vinfo = NULL;
    fd = open("/dev/graphics/fb0", O_RDONLY);
    if (fd < 0) 
    {
        printf("Fail to open frame buffer!\r\n");
        return -1;
    }
    rst = get_fb_fix_screeninfo(fd, &finfo);
    rst = get_fb_var_screeninfo(fd, &vinfo);
    printf("v0.5 type:%d; line_length:%d\r\n", finfo->type, finfo->line_length);
    printf("red: offset:%d len:%d type:%d\r\n", vinfo->red.offset, vinfo->red.length, vinfo->red.msb_right);
    printf("Gree: offset:%d, len:%d, type:%d\r\n", vinfo->green.offset, vinfo->green.length, vinfo->green.msb_right);
    printf("Blue: offset:%d, len:%d, type:%d\r\n", vinfo->blue.offset, vinfo->blue.length, vinfo->blue.msb_right);
    printf("A: offset:%d, len:%d, type:%d\r\n", vinfo->transp.offset, vinfo->transp.length, vinfo->transp.msb_right);
    printf("screen: width:%d, height:%d\r\n", vinfo->xres, vinfo->yres);
    printf("virtual screen: width:%d, height:%d\r\n", vinfo->xres_virtual, vinfo->yres_virtual);
    printf("virtual offset: xoffset:%d, yoffset:%d\r\n", vinfo->xoffset, vinfo->yoffset);
    free(finfo);
    free(vinfo);
    close(fd);
    
    get_frame_buffer_rgb();
    return 0;
}

int get_fb_fix_screeninfo(int fd, struct fb_fix_screeninfo** finfo)
{
    int ret;
    *finfo = (struct fb_fix_screeninfo*)malloc(sizeof(struct fb_fix_screeninfo));
    ret = ioctl(fd, FBIOGET_FSCREENINFO, *finfo);
    if (ret < 0)
    {
        printf("Fail to get screen info!\r\n");
        return -2;
    }
    return 0;
}

int get_fb_var_screeninfo(int fd, struct fb_var_screeninfo** vinfo)
{
    int ret;
    *vinfo = (struct fb_var_screeninfo*)malloc(sizeof(struct fb_var_screeninfo));
    //static struct fb_var_screeninfo vinfo;
    // 打开Framebuffer设备  
    fd = open("/dev/graphics/fb0", O_RDONLY);
    // 获取FrameBuffer 的 variable info 可变信息  
    ret = ioctl(fd, FBIOGET_VSCREENINFO, *vinfo);  
    if(ret < 0 )  
    {  
        printf("======Cannot get variable screen information.");  
        close(fd);  
        return -2;  
    }  
    return 0;
}

int get_frame_buffer_rgb()
{
    FILE* p_fbf = NULL;
    int fbfd = 0;
    struct fb_var_screeninfo vinfo;
    struct fb_fix_screeninfo finfo;
    long int screensize = 0;
    char *fbp = 0;
    long int location = 0;
    unsigned char* fbuf = (unsigned char*)malloc(sizeof(unsigned char)*(1440*2560*4));
    size_t read_len = 0;
    // Open the file for reading and writing
    fbfd = open("/dev/graphics/fb0", O_RDWR);
    if (!fbfd) {
        printf("Error: cannot open framebuffer device.\n");
        return -1;
    }
    printf("The framebuffer device was opened successfully.\n");
    // Get fixed screen information
    if (ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo)) {
        printf("Error reading fixed information.\n");
        return -2;
    }
    // Get variable screen information
    if (ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo)) {
        printf("Error reading variable information.\n");
        return -3;
    }
    screensize =  finfo.smem_len;
    printf("x:%d, y:%d, p:%d smem_len=%d\r\n", vinfo.xres, vinfo.yres, vinfo.bits_per_pixel, finfo.smem_len);
    printf("Befor %d screensize=%ld\r\n", (int)fbp, screensize);
    // screensize = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel >> 3  // >>3 表示算出字节数
    fbp = (char *)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED,fbfd, 0);
    if ((int)fbp == -1) {
        printf("Error: failed to map framebuffer device to memory.\n");
    }
    printf("read fb len=%lu  %d\r\n", (unsigned long)sizeof(fbp), (int)fbp);
    close(fbfd);
    
    
    p_fbf = fopen("/dev/graphics/fb0", "rb");
    read_len = fread(fbuf, sizeof(unsigned char), 1440*2560*4, p_fbf);
    printf("frame buffer read_len=%zu\r\n", read_len);
    
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * vinfo.yres);
    for(int y = 0; y < vinfo.yres; y++) {
        row_pointers[y] = (png_byte*)malloc(sizeof(png_byte)*(vinfo.xres*4));
        for (int x=0; x<vinfo.xres; x++) {
            row_pointers[y][x*4] = fbuf[y*vinfo.xres*4 + x*4 + 3];
            row_pointers[y][x*4 + 1] = fbuf[y*vinfo.xres*4 + x*4 + 2];
            row_pointers[y][x*4 + 2] = fbuf[y*vinfo.xres*4 + x*4 + 1];
            row_pointers[y][x*4 + 3] = fbuf[y*vinfo.xres*4 + x*4];
        }
    }
    fclose(p_fbf);
    write_png_file_data("/system/bin/a01.png", row_pointers, vinfo.xres, vinfo.yres);
    for(int y = 0; y < vinfo.yres; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    
    
    return 0;
}

int t001() 
{
    unsigned char fb_row[1440*4];
    size_t len = 0;
    int read_len = 0;
    /*FILE* p_fbf = NULL;
    p_fbf = fopen("/dev/graphics/fb0", "rb");
    if (NULL == p_fbf)
    {
        printf("Fail to open frame buffer file\r\n");
        return -1;
    }
    fseek(p_fbf, 0, SEEK_SET);
    len = fread(fb_row, sizeof(unsigned char), 1440*4, p_fbf);
    printf("v0.0.1 frame buffer:%zu\r\n", len);
    fclose(p_fbf);*/
    int fd;
    if ((fd = open("/dev/graphics/fb0", O_RDONLY)) < 0)
    {
        printf("Fail to open frame buffer!\r\n");
        return -1;
    }
    read_len = read(fd, fb_row, 1440*4);
    printf("read_len=%d\r\n", read_len);
    printf("get frame buffer image data\r\n");
    return 0;
}