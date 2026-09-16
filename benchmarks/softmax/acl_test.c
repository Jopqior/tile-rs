/* Minimal ACL sanity check — does aclInit work at all in this environment? */
#include <acl/acl.h>
#include <stdio.h>

int main(void) {
    printf("[acl_test] calling aclInit(NULL)...\n");
    aclError ret = aclInit(NULL);
    printf("[acl_test] aclInit returned: %d\n", ret);
    if (ret != 0) {
        const char *msg = aclGetRecentErrMsg();
        if (msg) printf("[acl_test] %s\n", msg);
        return 1;
    }

    int32_t count = -1;
    ret = aclrtGetDeviceCount((uint32_t*)&count);
    printf("[acl_test] aclrtGetDeviceCount: ret=%d count=%d\n", ret, count);

    aclFinalize();
    return 0;
}
