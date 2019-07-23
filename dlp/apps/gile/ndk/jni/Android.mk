LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_C_INCLUDES := ${LOCAL_PATH}/include/png
LOCAL_EXPORT_LDLIBS := -lz
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/.
LOCAL_MODULE := gile

LOCAL_SRC_FILES := app_main.c \
                    udp_client.c \
                    png/png.c \
                    png/pngerror.c \
                    png/pngget.c \
                    png/pngmem.c \
                    png/pngpread.c \
                    png/pngread.c \
                    png/pngrio.c \
                    png/pngrtran.c \
                    png/pngrutil.c \
                    png/pngset.c \
                    png/pngtest.c \
                    png/pngtrans.c \
                    png/pngwio.c \
                    png/pngwrite.c \
                    png/pngwtran.c \
                    png/pngwutil.c \
                    png/arm/arm_init.c \
                    png/arm/filter_neon.S \
                    png/arm/filter_neon_intrinsics.c \
                    png_util.c \
                    screen_capture.c

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI), armeabi-v7a x86))
    LOCAL_CFLAGS := -DHAVE_NEON=1
ifeq ($(TARGET_ARCH_ABI),x86)
    LOCAL_CFLAGS += -mssse3
endif
    LOCAL_SRC_FILES += helloneon-intrinsics.c.neon
endif

LOCAL_STATIC_LIBRARIES := cpufeatures

LOCAL_LDLIBS := -llog -lz


include $(BUILD_EXECUTABLE)

$(call import-module,android/cpufeatures)
