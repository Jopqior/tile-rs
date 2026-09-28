use num_traits::FromPrimitive;
use std::fmt;

// FIXME: some of the error codes are missed.
#[repr(i32)]
#[derive(Debug, PartialEq, Eq, Clone, FromPrimitive)]
pub enum AclInnerError {
    AclSuccess = ascend_sys::core::ACL_SUCCESS,
    AclInnerErrorInvalidParam = ascend_sys::core::ACL_ERROR_INVALID_PARAM,
    AclInnerErrorUninitialize = ascend_sys::core::ACL_ERROR_UNINITIALIZE,
    AclInnerErrorRepeatInitialize = ascend_sys::core::ACL_ERROR_REPEAT_INITIALIZE,
    AclInnerErrorInvalidFile = ascend_sys::core::ACL_ERROR_INVALID_FILE,
    AclInnerErrorWriteFile = ascend_sys::core::ACL_ERROR_WRITE_FILE,
    AclInnerErrorInvalidFileSize = ascend_sys::core::ACL_ERROR_INVALID_FILE_SIZE,
    AclInnerErrorParseFile = ascend_sys::core::ACL_ERROR_PARSE_FILE,
    AclInnerErrorFileMissingAttr = ascend_sys::core::ACL_ERROR_FILE_MISSING_ATTR,
    AclInnerErrorFileAttrInvalid = ascend_sys::core::ACL_ERROR_FILE_ATTR_INVALID,
    AclInnerErrorInvalidDumpConfig = ascend_sys::core::ACL_ERROR_INVALID_DUMP_CONFIG,
    AclInnerErrorInvalidProfilingConfig = ascend_sys::core::ACL_ERROR_INVALID_PROFILING_CONFIG,
    AclInnerErrorInvalidModelId = ascend_sys::core::ACL_ERROR_INVALID_MODEL_ID,
    AclInnerErrorDeserializeModel = ascend_sys::core::ACL_ERROR_DESERIALIZE_MODEL,
    AclInnerErrorParseModel = ascend_sys::core::ACL_ERROR_PARSE_MODEL,
    AclInnerErrorReadModelFailure = ascend_sys::core::ACL_ERROR_READ_MODEL_FAILURE,
    AclInnerErrorModelSizeInvalid = ascend_sys::core::ACL_ERROR_MODEL_SIZE_INVALID,
    AclInnerErrorModelMissingAttr = ascend_sys::core::ACL_ERROR_MODEL_MISSING_ATTR,
    AclInnerErrorModelInputNotMatch = ascend_sys::core::ACL_ERROR_MODEL_INPUT_NOT_MATCH,
    AclInnerErrorModelOutputNotMatch = ascend_sys::core::ACL_ERROR_MODEL_OUTPUT_NOT_MATCH,
    AclInnerErrorModelNotDynamic = ascend_sys::core::ACL_ERROR_MODEL_NOT_DYNAMIC,
    AclInnerErrorOpTypeNotMatch = ascend_sys::core::ACL_ERROR_OP_TYPE_NOT_MATCH,
    AclInnerErrorOpInputNotMatch = ascend_sys::core::ACL_ERROR_OP_INPUT_NOT_MATCH,
    AclInnerErrorOpOutputNotMatch = ascend_sys::core::ACL_ERROR_OP_OUTPUT_NOT_MATCH,
    AclInnerErrorOpAttrNotMatch = ascend_sys::core::ACL_ERROR_OP_ATTR_NOT_MATCH,
    AclInnerErrorOpNotFound = ascend_sys::core::ACL_ERROR_OP_NOT_FOUND,
    AclInnerErrorOpLoadFailed = ascend_sys::core::ACL_ERROR_OP_LOAD_FAILED,
    AclInnerErrorUnsupportedDataType = ascend_sys::core::ACL_ERROR_UNSUPPORTED_DATA_TYPE,
    AclInnerErrorFormatNotMatch = ascend_sys::core::ACL_ERROR_FORMAT_NOT_MATCH,
    AclInnerErrorBinSelectorNotRegistered = ascend_sys::core::ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED,
    AclInnerErrorKernelNotFound = ascend_sys::core::ACL_ERROR_KERNEL_NOT_FOUND,
    AclInnerErrorBinSelectorAlreadyRegistered =
        ascend_sys::core::ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED,
    AclInnerErrorKernelAlreadyRegistered = ascend_sys::core::ACL_ERROR_KERNEL_ALREADY_REGISTERED,
    AclInnerErrorInvalidQueueId = ascend_sys::core::ACL_ERROR_INVALID_QUEUE_ID,
    AclInnerErrorRepeatSubscribe = ascend_sys::core::ACL_ERROR_REPEAT_SUBSCRIBE,
    AclInnerErrorRepeatFinalize = ascend_sys::core::ACL_ERROR_REPEAT_FINALIZE,
    AclInnerErrorCompilingStubMode = ascend_sys::core::ACL_ERROR_COMPILING_STUB_MODE,
    AclInnerErrorProfAlreadyRun = ascend_sys::core::ACL_ERROR_PROF_ALREADY_RUN,
    AclInnerErrorProfNotRun = ascend_sys::core::ACL_ERROR_PROF_NOT_RUN,
    AclInnerErrorDumpAlreadyRun = ascend_sys::core::ACL_ERROR_DUMP_ALREADY_RUN,
    AclInnerErrorDumpNotRun = ascend_sys::core::ACL_ERROR_DUMP_NOT_RUN,
    AclInnerErrorProfRepeatSubscribe = ascend_sys::core::ACL_ERROR_PROF_REPEAT_SUBSCRIBE,
    AclInnerErrorProfApiConflict = ascend_sys::core::ACL_ERROR_PROF_API_CONFLICT,
    AclInnerErrorInvalidMaxOpqueueNumConfig =
        ascend_sys::core::ACL_ERROR_INVALID_MAX_OPQUEUE_NUM_CONFIG,
    AclInnerErrorInvalidOppPath = ascend_sys::core::ACL_ERROR_INVALID_OPP_PATH,
    AclInnerErrorOpUnsupportedDynamic = ascend_sys::core::ACL_ERROR_OP_UNSUPPORTED_DYNAMIC,
    AclInnerErrorRelativeResourceNotCleared =
        ascend_sys::core::ACL_ERROR_RELATIVE_RESOURCE_NOT_CLEARED,
    AclInnerErrorUnsupportedJpeg = ascend_sys::core::ACL_ERROR_UNSUPPORTED_JPEG,
    AclInnerErrorBadAlloc = ascend_sys::core::ACL_ERROR_BAD_ALLOC,
    AclInnerErrorApiNotSupport = ascend_sys::core::ACL_ERROR_API_NOT_SUPPORT,
    AclInnerErrorMemoryAddressUnaligned = ascend_sys::core::ACL_ERROR_MEMORY_ADDRESS_UNALIGNED,
    AclInnerErrorResourceNotMatch = ascend_sys::core::ACL_ERROR_RESOURCE_NOT_MATCH,
    AclInnerErrorInvalidResourceHandle = ascend_sys::core::ACL_ERROR_INVALID_RESOURCE_HANDLE,
    AclInnerErrorFeatureUnsupported = ascend_sys::core::ACL_ERROR_FEATURE_UNSUPPORTED,
    AclInnerErrorProfModulesUnsupported = ascend_sys::core::ACL_ERROR_PROF_MODULES_UNSUPPORTED,
    AclInnerErrorStorageOverLimit = ascend_sys::core::ACL_ERROR_STORAGE_OVER_LIMIT,
    AclInnerErrorInternalError = ascend_sys::core::ACL_ERROR_INTERNAL_ERROR,
    AclInnerErrorFailure = ascend_sys::core::ACL_ERROR_FAILURE,
    AclInnerErrorGeFailure = ascend_sys::core::ACL_ERROR_GE_FAILURE,
    AclInnerErrorRtFailure = ascend_sys::core::ACL_ERROR_RT_FAILURE,
    AclInnerErrorDrvFailure = ascend_sys::core::ACL_ERROR_DRV_FAILURE,
    AclInnerErrorProfilingFailure = ascend_sys::core::ACL_ERROR_PROFILING_FAILURE,
    // Runtime error codes
    AclInnerErrorRtParamInvalid = ascend_sys::core::ACL_ERROR_RT_PARAM_INVALID as i32,
    AclInnerErrorRtInvalidDeviceId = ascend_sys::core::ACL_ERROR_RT_INVALID_DEVICEID as i32,
    AclInnerErrorRtContextNull = ascend_sys::core::ACL_ERROR_RT_CONTEXT_NULL as i32,
    AclInnerErrorRtStreamContext = ascend_sys::core::ACL_ERROR_RT_STREAM_CONTEXT as i32,
    AclInnerErrorRtModelContext = ascend_sys::core::ACL_ERROR_RT_MODEL_CONTEXT as i32,
    AclInnerErrorRtStreamModel = ascend_sys::core::ACL_ERROR_RT_STREAM_MODEL as i32,
    AclInnerErrorRtEventTimestampInvalid =
        ascend_sys::core::ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID as i32,
    AclInnerErrorRtEventTimestampReversal =
        ascend_sys::core::ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL as i32,
    AclInnerErrorRtAddrUnaligned = ascend_sys::core::ACL_ERROR_RT_ADDR_UNALIGNED as i32,
    AclInnerErrorRtFileOpen = ascend_sys::core::ACL_ERROR_RT_FILE_OPEN as i32,
    AclInnerErrorRtFileWrite = ascend_sys::core::ACL_ERROR_RT_FILE_WRITE as i32,
    AclInnerErrorRtStreamSubscribe = ascend_sys::core::ACL_ERROR_RT_STREAM_SUBSCRIBE as i32,
    AclInnerErrorRtThreadSubscribe = ascend_sys::core::ACL_ERROR_RT_THREAD_SUBSCRIBE as i32,
    AclInnerErrorRtGroupNotSet = ascend_sys::core::ACL_ERROR_RT_GROUP_NOT_SET as i32,
    AclInnerErrorRtGroupNotCreate = ascend_sys::core::ACL_ERROR_RT_GROUP_NOT_CREATE as i32,
    AclInnerErrorRtStreamNoCbReg = ascend_sys::core::ACL_ERROR_RT_STREAM_NO_CB_REG as i32,
    AclInnerErrorRtInvalidMemoryType = ascend_sys::core::ACL_ERROR_RT_INVALID_MEMORY_TYPE as i32,
    AclInnerErrorRtInvalidHandle = ascend_sys::core::ACL_ERROR_RT_INVALID_HANDLE as i32,
    AclInnerErrorRtInvalidMallocType = ascend_sys::core::ACL_ERROR_RT_INVALID_MALLOC_TYPE as i32,
    AclInnerErrorRtWaitTimeout = ascend_sys::core::ACL_ERROR_RT_WAIT_TIMEOUT as i32,
    AclInnerErrorRtTaskTimeout = ascend_sys::core::ACL_ERROR_RT_TASK_TIMEOUT as i32,
    AclInnerErrorRtSysparamoptNotSet = ascend_sys::core::ACL_ERROR_RT_SYSPARAMOPT_NOT_SET as i32,
    AclInnerErrorRtFeatureNotSupport = ascend_sys::core::ACL_ERROR_RT_FEATURE_NOT_SUPPORT as i32,
    AclInnerErrorRtMemoryAllocation = ascend_sys::core::ACL_ERROR_RT_MEMORY_ALLOCATION as i32,
    AclInnerErrorRtMemoryFree = ascend_sys::core::ACL_ERROR_RT_MEMORY_FREE as i32,
    AclInnerErrorRtAicoreOverFlow = ascend_sys::core::ACL_ERROR_RT_AICORE_OVER_FLOW as i32,
    AclInnerErrorRtNoDevice = ascend_sys::core::ACL_ERROR_RT_NO_DEVICE as i32,
    AclInnerErrorRtResourceAllocFail = ascend_sys::core::ACL_ERROR_RT_RESOURCE_ALLOC_FAIL as i32,
    AclInnerErrorRtNoPermission = ascend_sys::core::ACL_ERROR_RT_NO_PERMISSION as i32,
    AclInnerErrorRtNoEventResource = ascend_sys::core::ACL_ERROR_RT_NO_EVENT_RESOURCE as i32,
    AclInnerErrorRtNoStreamResource = ascend_sys::core::ACL_ERROR_RT_NO_STREAM_RESOURCE as i32,
    AclInnerErrorRtNoNotifyResource = ascend_sys::core::ACL_ERROR_RT_NO_NOTIFY_RESOURCE as i32,
    AclInnerErrorRtNoModelResource = ascend_sys::core::ACL_ERROR_RT_NO_MODEL_RESOURCE as i32,
    AclInnerErrorRtNoCdqResource = ascend_sys::core::ACL_ERROR_RT_NO_CDQ_RESOURCE as i32,
    AclInnerErrorRtOverLimit = ascend_sys::core::ACL_ERROR_RT_OVER_LIMIT as i32,
    AclInnerErrorRtQueueEmpty = ascend_sys::core::ACL_ERROR_RT_QUEUE_EMPTY as i32,
    AclInnerErrorRtQueueFull = ascend_sys::core::ACL_ERROR_RT_QUEUE_FULL as i32,
    AclInnerErrorRtRepeatedInit = ascend_sys::core::ACL_ERROR_RT_REPEATED_INIT as i32,
    AclInnerErrorRtDeviceOom = ascend_sys::core::ACL_ERROR_RT_DEVICE_OOM as i32,
    AclInnerErrorRtInternalError = ascend_sys::core::ACL_ERROR_RT_INTERNAL_ERROR as i32,
    AclInnerErrorRtTsError = ascend_sys::core::ACL_ERROR_RT_TS_ERROR as i32,
    AclInnerErrorRtStreamTaskFull = ascend_sys::core::ACL_ERROR_RT_STREAM_TASK_FULL as i32,
    AclInnerErrorRtStreamTaskEmpty = ascend_sys::core::ACL_ERROR_RT_STREAM_TASK_EMPTY as i32,
    AclInnerErrorRtStreamNotComplete = ascend_sys::core::ACL_ERROR_RT_STREAM_NOT_COMPLETE as i32,
    AclInnerErrorRtEndOfSequence = ascend_sys::core::ACL_ERROR_RT_END_OF_SEQUENCE as i32,
    AclInnerErrorRtEventNotComplete = ascend_sys::core::ACL_ERROR_RT_EVENT_NOT_COMPLETE as i32,
    AclInnerErrorRtContextReleaseError =
        ascend_sys::core::ACL_ERROR_RT_CONTEXT_RELEASE_ERROR as i32,
    AclInnerErrorRtSocVersion = ascend_sys::core::ACL_ERROR_RT_SOC_VERSION as i32,
    AclInnerErrorRtTaskTypeNotSupport = ascend_sys::core::ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT as i32,
    AclInnerErrorRtLostHeartbeat = ascend_sys::core::ACL_ERROR_RT_LOST_HEARTBEAT as i32,
    AclInnerErrorRtModelExecute = ascend_sys::core::ACL_ERROR_RT_MODEL_EXECUTE as i32,
    AclInnerErrorRtReportTimeout = ascend_sys::core::ACL_ERROR_RT_REPORT_TIMEOUT as i32,
    AclInnerErrorRtSysDma = ascend_sys::core::ACL_ERROR_RT_SYS_DMA as i32,
    AclInnerErrorRtAicoreTimeout = ascend_sys::core::ACL_ERROR_RT_AICORE_TIMEOUT as i32,
    AclInnerErrorRtAicoreException = ascend_sys::core::ACL_ERROR_RT_AICORE_EXCEPTION as i32,
    AclInnerErrorRtAicoreTrapException =
        ascend_sys::core::ACL_ERROR_RT_AICORE_TRAP_EXCEPTION as i32,
    AclInnerErrorRtAicpuTimeout = ascend_sys::core::ACL_ERROR_RT_AICPU_TIMEOUT as i32,
    AclInnerErrorRtAicpuException = ascend_sys::core::ACL_ERROR_RT_AICPU_EXCEPTION as i32,
    AclInnerErrorRtAicpuDatadumpRspErr =
        ascend_sys::core::ACL_ERROR_RT_AICPU_DATADUMP_RSP_ERR as i32,
    AclInnerErrorRtAicpuModelRspErr = ascend_sys::core::ACL_ERROR_RT_AICPU_MODEL_RSP_ERR as i32,
    AclInnerErrorRtProfilingError = ascend_sys::core::ACL_ERROR_RT_PROFILING_ERROR as i32,
    AclInnerErrorRtIpcError = ascend_sys::core::ACL_ERROR_RT_IPC_ERROR as i32,
    AclInnerErrorRtModelAbortNormal = ascend_sys::core::ACL_ERROR_RT_MODEL_ABORT_NORMAL as i32,
    AclInnerErrorRtKernelUnregistering = ascend_sys::core::ACL_ERROR_RT_KERNEL_UNREGISTERING as i32,
    AclInnerErrorRtRingbufferNotInit = ascend_sys::core::ACL_ERROR_RT_RINGBUFFER_NOT_INIT as i32,
    AclInnerErrorRtRingbufferNoData = ascend_sys::core::ACL_ERROR_RT_RINGBUFFER_NO_DATA as i32,
    AclInnerErrorRtKernelLookup = ascend_sys::core::ACL_ERROR_RT_KERNEL_LOOKUP as i32,
    AclInnerErrorRtKernelDuplicate = ascend_sys::core::ACL_ERROR_RT_KERNEL_DUPLICATE as i32,
    AclInnerErrorRtDebugRegisterFail = ascend_sys::core::ACL_ERROR_RT_DEBUG_REGISTER_FAIL as i32,
    AclInnerErrorRtDebugUnregisterFail =
        ascend_sys::core::ACL_ERROR_RT_DEBUG_UNREGISTER_FAIL as i32,
    AclInnerErrorRtLabelContext = ascend_sys::core::ACL_ERROR_RT_LABEL_CONTEXT as i32,
    AclInnerErrorRtProgramUseOut = ascend_sys::core::ACL_ERROR_RT_PROGRAM_USE_OUT as i32,
    AclInnerErrorRtDevSetupError = ascend_sys::core::ACL_ERROR_RT_DEV_SETUP_ERROR as i32,
    AclInnerErrorRtVectorCoreTimeout = ascend_sys::core::ACL_ERROR_RT_VECTOR_CORE_TIMEOUT as i32,
    AclInnerErrorRtVectorCoreException =
        ascend_sys::core::ACL_ERROR_RT_VECTOR_CORE_EXCEPTION as i32,
    AclInnerErrorRtVectorCoreTrapException =
        ascend_sys::core::ACL_ERROR_RT_VECTOR_CORE_TRAP_EXCEPTION as i32,
    AclInnerErrorRtCdqBatchAbnormal = ascend_sys::core::ACL_ERROR_RT_CDQ_BATCH_ABNORMAL as i32,
    AclInnerErrorRtDieModeChangeError = ascend_sys::core::ACL_ERROR_RT_DIE_MODE_CHANGE_ERROR as i32,
    AclInnerErrorRtDieSetError = ascend_sys::core::ACL_ERROR_RT_DIE_SET_ERROR as i32,
    AclInnerErrorRtInvalidDieid = ascend_sys::core::ACL_ERROR_RT_INVALID_DIEID as i32,
    AclInnerErrorRtDieModeNotSet = ascend_sys::core::ACL_ERROR_RT_DIE_MODE_NOT_SET as i32,
    AclInnerErrorRtAicoreTrapReadOverflow =
        ascend_sys::core::ACL_ERROR_RT_AICORE_TRAP_READ_OVERFLOW as i32,
    AclInnerErrorRtAicoreTrapWriteOverflow =
        ascend_sys::core::ACL_ERROR_RT_AICORE_TRAP_WRITE_OVERFLOW as i32,
    AclInnerErrorRtVectorCoreTrapReadOverflow =
        ascend_sys::core::ACL_ERROR_RT_VECTOR_CORE_TRAP_READ_OVERFLOW as i32,
    AclInnerErrorRtVectorCoreTrapWriteOverflow =
        ascend_sys::core::ACL_ERROR_RT_VECTOR_CORE_TRAP_WRITE_OVERFLOW as i32,
    AclInnerErrorRtStreamSyncTimeout = ascend_sys::core::ACL_ERROR_RT_STREAM_SYNC_TIMEOUT as i32,
    AclInnerErrorRtEventSyncTimeout = ascend_sys::core::ACL_ERROR_RT_EVENT_SYNC_TIMEOUT as i32,
    AclInnerErrorRtFftsPlusTimeout = ascend_sys::core::ACL_ERROR_RT_FFTS_PLUS_TIMEOUT as i32,
    AclInnerErrorRtFftsPlusException = ascend_sys::core::ACL_ERROR_RT_FFTS_PLUS_EXCEPTION as i32,
    AclInnerErrorRtFftsPlusTrapException =
        ascend_sys::core::ACL_ERROR_RT_FFTS_PLUS_TRAP_EXCEPTION as i32,
    AclInnerErrorRtSendMsg = ascend_sys::core::ACL_ERROR_RT_SEND_MSG as i32,
    AclInnerErrorRtCopyData = ascend_sys::core::ACL_ERROR_RT_COPY_DATA as i32,
    AclInnerErrorRtDrvInternalError = ascend_sys::core::ACL_ERROR_RT_DRV_INTERNAL_ERROR as i32,
    AclInnerErrorRtAicpuInternalError = ascend_sys::core::ACL_ERROR_RT_AICPU_INTERNAL_ERROR as i32,
    AclInnerErrorRtSocketClose = ascend_sys::core::ACL_ERROR_RT_SOCKET_CLOSE as i32,
    AclInnerErrorRtAicpuInfoLoadRspErr =
        ascend_sys::core::ACL_ERROR_RT_AICPU_INFO_LOAD_RSP_ERR as i32,
    // GE error codes from documentation
    AclInnerErrorGeParamInvalid = ascend_sys::core::ACL_ERROR_GE_PARAM_INVALID as i32,
    AclInnerErrorGeExecNotInit = ascend_sys::core::ACL_ERROR_GE_EXEC_NOT_INIT as i32,
    AclInnerErrorGeExecModelPathInvalid =
        ascend_sys::core::ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID as i32,
    AclInnerErrorGeExecModelIdInvalid = ascend_sys::core::ACL_ERROR_GE_EXEC_MODEL_ID_INVALID as i32,
    AclInnerErrorGeExecModelDataSizeInvalid =
        ascend_sys::core::ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID as i32,
    AclInnerErrorGeExecModelAddrInvalid =
        ascend_sys::core::ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID as i32,
    AclInnerErrorGeExecModelQueueIdInvalid =
        ascend_sys::core::ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID as i32,
    AclInnerErrorGeExecLoadModelRepeated =
        ascend_sys::core::ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED as i32,
    AclInnerErrorGeDynamicInputAddrInvalid =
        ascend_sys::core::ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID as i32,
    AclInnerErrorGeDynamicInputLengthInvalid =
        ascend_sys::core::ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID as i32,
    AclInnerErrorGeDynamicBatchSizeInvalid =
        ascend_sys::core::ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID as i32,
    AclInnerErrorGeAippBatchEmpty = ascend_sys::core::ACL_ERROR_GE_AIPP_BATCH_EMPTY as i32,
    AclInnerErrorGeAippNotExist = ascend_sys::core::ACL_ERROR_GE_AIPP_NOT_EXIST as i32,
    AclInnerErrorGeAippModeInvalid = ascend_sys::core::ACL_ERROR_GE_AIPP_MODE_INVALID as i32,
    AclInnerErrorGeOpTaskTypeInvalid = ascend_sys::core::ACL_ERROR_GE_OP_TASK_TYPE_INVALID as i32,
    AclInnerErrorGeOpKernelTypeInvalid =
        ascend_sys::core::ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID as i32,
    AclInnerErrorGePlgmgrPathInvalid = ascend_sys::core::ACL_ERROR_GE_PLGMGR_PATH_INVALID as i32,
    AclInnerErrorGeFormatInvalid = ascend_sys::core::ACL_ERROR_GE_FORMAT_INVALID as i32,
    AclInnerErrorGeShapeInvalid = ascend_sys::core::ACL_ERROR_GE_SHAPE_INVALID as i32,
    AclInnerErrorGeDatatypeInvalid = ascend_sys::core::ACL_ERROR_GE_DATATYPE_INVALID as i32,
    AclInnerErrorGeMemoryAllocation = ascend_sys::core::ACL_ERROR_GE_MEMORY_ALLOCATION as i32,
    AclInnerErrorGeMemoryOperateFailed =
        ascend_sys::core::ACL_ERROR_GE_MEMORY_OPERATE_FAILED as i32,
    AclInnerErrorGeDeviceMemoryAllocationFailed =
        ascend_sys::core::ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED as i32,
    AclInnerErrorGeUserRaiseException = ascend_sys::core::ACL_ERROR_GE_USER_RAISE_EXCEPTION as i32,
    AclInnerErrorGeDataNotAligned = ascend_sys::core::ACL_ERROR_GE_DATA_NOT_ALIGNED as i32,
    AclInnerErrorGeInternalError = ascend_sys::core::ACL_ERROR_GE_INTERNAL_ERROR as i32,
    AclInnerErrorGeLoadModel = ascend_sys::core::ACL_ERROR_GE_LOAD_MODEL as i32,
    AclInnerErrorGeExecLoadModelPartitionFailed =
        ascend_sys::core::ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED as i32,
    AclInnerErrorGeExecLoadWeightPartitionFailed =
        ascend_sys::core::ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED as i32,
    AclInnerErrorGeExecLoadTaskPartitionFailed =
        ascend_sys::core::ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED as i32,
    AclInnerErrorGeExecLoadKernelPartitionFailed =
        ascend_sys::core::ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED as i32,
    AclInnerErrorGeExecReleaseModelData =
        ascend_sys::core::ACL_ERROR_GE_EXEC_RELEASE_MODEL_DATA as i32,
    AclInnerErrorGeCommandHandle = ascend_sys::core::ACL_ERROR_GE_COMMAND_HANDLE as i32,
    AclInnerErrorGeGetTensorInfo = ascend_sys::core::ACL_ERROR_GE_GET_TENSOR_INFO as i32,
    AclInnerErrorGeUnloadModel = ascend_sys::core::ACL_ERROR_GE_UNLOAD_MODEL as i32,
    AclInnerErrorGeModelExecuteTimeout =
        ascend_sys::core::ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT as i32,
}

impl fmt::Display for AclInnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AclInnerError::AclSuccess => write!(f, "Execution successful."),
            AclInnerError::AclInnerErrorInvalidParam => write!(
                f,
                "Parameter validation failed.\nPlease check whether the input parameter value of the interface is correct."
            ),
            AclInnerError::AclInnerErrorUninitialize => write!(
                f,
                "ACL is not initialized.\nPlease check whether the acl.init interface has been called for initialization. Please make sure that the acl.init interface has been called and that it is called before other pyACL interfaces.\nPlease check whether the initialization interface of the corresponding function has been called, such as the acl.mdl.init_dump interface for initializing Dump and the acl.prof.init interface for initializing Profiling."
            ),
            AclInnerError::AclInnerErrorRepeatInitialize => write!(
                f,
                "Repeat initialization or repeat loading.\nPlease check whether the corresponding interface is called to initialize or load repeatedly."
            ),
            AclInnerError::AclInnerErrorInvalidFile => write!(
                f,
                "Invalid file.\nPlease check whether the file exists, whether the file can be accessed, etc."
            ),
            AclInnerError::AclInnerErrorWriteFile => write!(
                f,
                "Failed to write file.\nPlease check whether the file path exists and whether the file has write permission."
            ),
            AclInnerError::AclInnerErrorInvalidFileSize => write!(
                f,
                "Invalid file size.\nPlease check whether the file size meets the interface requirements."
            ),
            AclInnerError::AclInnerErrorParseFile => write!(
                f,
                "Failed to parse the file.\nPlease check whether the file content is legal."
            ),
            AclInnerError::AclInnerErrorFileMissingAttr => write!(
                f,
                "File is missing parameters.\nPlease check that the file content is complete."
            ),
            AclInnerError::AclInnerErrorFileAttrInvalid => write!(
                f,
                "The file parameters are invalid.\nPlease check whether the parameter values ​​in the file are correct."
            ),
            AclInnerError::AclInnerErrorInvalidDumpConfig => write!(
                f,
                "Invalid dump configuration.\nPlease check whether the Dump configuration is correct. For detailed configuration, see\u{00a0}the Precision Debugging Tool Guide\u{00a0}."
            ),
            AclInnerError::AclInnerErrorInvalidProfilingConfig => write!(
                f,
                "Invalid Profiling configuration.\nPlease check whether the Profiling configuration is correct."
            ),
            AclInnerError::AclInnerErrorInvalidModelId => write!(
                f,
                "Invalid model ID.\nPlease check whether the model ID is correct and whether the model is loaded correctly."
            ),
            AclInnerError::AclInnerErrorDeserializeModel => write!(
                f,
                "Failed to deserialize model.\nThe model may not match the current version, please rebuild the model."
            ),
            AclInnerError::AclInnerErrorParseModel => write!(
                f,
                "Failed to parse the model.\nThe model may not match the current version, please rebuild the model."
            ),
            AclInnerError::AclInnerErrorReadModelFailure => write!(
                f,
                "Failed to read model.\nPlease check whether the model file exists, whether the model file can be accessed, etc."
            ),
            AclInnerError::AclInnerErrorModelSizeInvalid => write!(
                f,
                "Invalid model size.\nThe model file is invalid, please rebuild the model."
            ),
            AclInnerError::AclInnerErrorModelMissingAttr => write!(
                f,
                "Model is missing parameters.\nThe model may not match the current version, please rebuild the model."
            ),
            AclInnerError::AclInnerErrorModelInputNotMatch => write!(
                f,
                "The input to the model does not match.\nPlease check that the model input is correct."
            ),
            AclInnerError::AclInnerErrorModelOutputNotMatch => write!(
                f,
                "The output of the model does not match.\nPlease check that the output of the model is correct."
            ),
            AclInnerError::AclInnerErrorModelNotDynamic => write!(
                f,
                "Non-dynamic model.\nPlease check whether the current model supports dynamic scenes. If not, please rebuild the model."
            ),
            AclInnerError::AclInnerErrorOpTypeNotMatch => write!(
                f,
                "Single operator type mismatch.\nPlease check whether the operator type is correct."
            ),
            AclInnerError::AclInnerErrorOpInputNotMatch => write!(
                f,
                "The input of a single operator does not match.\nPlease check whether the operator input is correct."
            ),
            AclInnerError::AclInnerErrorOpOutputNotMatch => write!(
                f,
                "The output of a single operator does not match.\nPlease check whether the output of the operator is correct."
            ),
            AclInnerError::AclInnerErrorOpAttrNotMatch => write!(
                f,
                "The properties of the single operator do not match.\nPlease check whether the properties of the operator are correct."
            ),
            AclInnerError::AclInnerErrorOpNotFound => write!(
                f,
                "Single operator not found.\nPlease check whether the operator type is supported."
            ),
            AclInnerError::AclInnerErrorOpLoadFailed => write!(
                f,
                "Single operator loading failed.\nThe model may not match the current version. Please rebuild the single operator model."
            ),
            AclInnerError::AclInnerErrorUnsupportedDataType => write!(
                f,
                "Unsupported data type.\nPlease check whether the data type exists or is currently supported."
            ),
            AclInnerError::AclInnerErrorFormatNotMatch => write!(
                f,
                "Format does not match.\nPlease check whether the Format is correct."
            ),
            AclInnerError::AclInnerErrorBinSelectorNotRegistered => write!(
                f,
                "When the operator interface is compiled using the binary selection method, the operator does not register the selector.\nPlease check whether the acl.op.register_compile_func interface is called to register the operator selector."
            ),
            AclInnerError::AclInnerErrorKernelNotFound => write!(
                f,
                "When compiling an operator, the operator Kernel is not registered.\nPlease check whether the acl.op.create_kernel interface is called to register the operator Kernel."
            ),
            AclInnerError::AclInnerErrorBinSelectorAlreadyRegistered => write!(
                f,
                "When the operator interface is compiled using the binary selection method, the operator is registered repeatedly.\nPlease check whether the acl.op.register_compile_func interface is called repeatedly to register the operator selector."
            ),
            AclInnerError::AclInnerErrorKernelAlreadyRegistered => write!(
                f,
                "When compiling an operator, the operator Kernel is registered repeatedly.\nPlease check whether the acl.op.create_kernel interface is called repeatedly to register the operator Kernel."
            ),
            AclInnerError::AclInnerErrorInvalidQueueId => write!(
                f,
                "Invalid queue ID.\nPlease check whether the queue ID is correct."
            ),
            AclInnerError::AclInnerErrorRepeatSubscribe => write!(
                f,
                "Repeat subscription.\nPlease check whether the acl.rt.subscribe_report\u{00a0}interface is called repeatedly for the same stream\u{00a0}."
            ),
            // Display implementations for new error variants
            AclInnerError::AclInnerErrorRepeatFinalize => write!(
                f,
                "Repeat deinitialization.\nPlease check whether the acl.finalize interface is called repeatedly for deinitialization."
            ),
            AclInnerError::AclInnerErrorCompilingStubMode => write!(
                f,
                "The dynamic library path configured before running the application is the path of the compiled stub, not the correct dynamic library path.\nPlease check the configuration of the dynamic library path and make sure to use the dynamic library of the running mode."
            ),
            AclInnerError::AclInnerErrorProfAlreadyRun => write!(
                f,
                "A task for collecting profiling data already exists.\nPlease check the code logic. The configuration of \"collecting profiling data by calling pyACL API\" cannot coexist with other profiling configurations. Only one can be retained. Check whether the profiling configuration is delivered to the same device multiple times."
            ),
            AclInnerError::AclInnerErrorProfNotRun => write!(
                f,
                "The acl.prof.init interface is not used to initialize Profiling.\nPlease check the order of calling the interfaces for collecting profiling data."
            ),
            AclInnerError::AclInnerErrorDumpAlreadyRun => write!(
                f,
                "A task for obtaining dump data already exists.\nPlease check whether the acl.init interface has been called to configure the Dump information before calling the acl.mdl.init_dump interface, acl.mdl.set_dump interface, and acl.mdl.finalize_dump interface to configure the Dump information. If so, please adjust the code logic and keep one way to configure the Dump information."
            ),
            AclInnerError::AclInnerErrorDumpNotRun => write!(
                f,
                "The acl.mdl.init_dump interface is not used to initialize the dump first.\nPlease check the interface calling sequence for obtaining Dump data, refer to the description of acl.mdl.init_dump interface."
            ),
            AclInnerError::AclInnerErrorProfRepeatSubscribe => write!(
                f,
                "Repeatedly subscribing to the same model.\nPlease check the order of API calls."
            ),
            AclInnerError::AclInnerErrorProfApiConflict => write!(
                f,
                "The interface calls for collecting performance data conflict.\nThe two modes of Profiling performance data collection interfaces cannot be called crosswise. The acl.prof.model_subscribe interface, acl.prof.get_op_* interface, and acl.prof.model_unsubscribe interface cannot be called between the acl.prof.init interface and the acl.prof.finalize interface. The acl.prof.init interface, acl.prof.start interface, acl.prof.stop interface, and acl.prof.finalize interface cannot be called between the acl.prof.model_subscribe interface and the acl.prof.model_unsubscribe interface."
            ),
            AclInnerError::AclInnerErrorInvalidMaxOpqueueNumConfig => write!(
                f,
                "Invalid operator cache information aging configuration.\nPlease check the operator cache information aging configuration, refer to the configuration instructions and examples in acl.init."
            ),
            AclInnerError::AclInnerErrorInvalidOppPath => write!(
                f,
                "The ASCEND_OPP_PATH environment variable is not set, or the value of the environment variable is set incorrectly.\nPlease check whether the ASCEND_OPP_PATH environment variable is set and whether the value of the environment variable is the installation path of the opp software package."
            ),
            AclInnerError::AclInnerErrorOpUnsupportedDynamic => write!(
                f,
                "The operator does not support dynamic shapes.\nPlease check whether the Shape of the operator in the single operator model file is dynamic. If it is dynamic, it needs to be changed to a fixed Shape. Please check whether the shape of aclTensorDesc is dynamic when compiling the operator. If it is dynamic, you need to recreate aclTensorDesc according to the fixed shape."
            ),
            AclInnerError::AclInnerErrorRelativeResourceNotCleared => write!(
                f,
                "The associated resources have not yet been released.\nWhen destroying channel description information, if the related channel has not been destroyed, this error code is returned. Please check whether the channel associated with this channel description information has been destroyed."
            ),
            AclInnerError::AclInnerErrorUnsupportedJpeg => write!(
                f,
                "The JPEGD function does not support the input image coding format (for example, arithmetic coding, progressive coding, etc.).\nWhen implementing the JPEGD picture decoding function, only Huffman coding is supported. The color space of the original image before compression is YUV, and the ratio of each component of the pixel is 4:4:4 or 4:2:2 or 4:2:0 or 4:0:0 or 4:4:0. Arithmetic coding, progressive JPEG format and JPEG2000 format are not supported."
            ),
            AclInnerError::AclInnerErrorBadAlloc => write!(
                f,
                "Failed to apply for memory.\nPlease check the remaining memory on your hardware environment."
            ),
            AclInnerError::AclInnerErrorApiNotSupport => write!(
                f,
                "The interface is not supported.\nPlease check whether the called interface is currently supported."
            ),
            AclInnerError::AclInnerErrorMemoryAddressUnaligned => write!(
                f,
                "Memory address is misaligned.\nPlease check whether the memory address meets the interface requirements."
            ),
            AclInnerError::AclInnerErrorResourceNotMatch => write!(
                f,
                "Resource mismatch.\nPlease check whether the correct Stream, Context and other resources are passed in when calling the interface."
            ),
            AclInnerError::AclInnerErrorInvalidResourceHandle => write!(
                f,
                "Invalid resource handle.\nPlease check whether the Stream, Context and other resources passed in when calling the interface have been destroyed or occupied."
            ),
            AclInnerError::AclInnerErrorFeatureUnsupported => write!(
                f,
                "Feature not supported.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorProfModulesUnsupported => write!(
                f,
                "An unsupported profiling configuration was delivered.\nRefer to the description in acl.prof.create_config to check whether the Profiling configuration is correct."
            ),
            AclInnerError::AclInnerErrorStorageOverLimit => write!(
                f,
                "Storage limit exceeded.\nPlease check the remaining storage space on your hardware environment."
            ),
            AclInnerError::AclInnerErrorInternalError => write!(
                f,
                "Unknown internal error.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorFailure => {
                write!(f, "An error occurred in the system's internal ACL.")
            }
            AclInnerError::AclInnerErrorGeFailure => write!(f, "GE error within the system."),
            AclInnerError::AclInnerErrorRtFailure => {
                write!(f, "Internal RUNTIME error in the system.")
            }
            AclInnerError::AclInnerErrorDrvFailure => {
                write!(f, "Internal DRV (Driver) error in the system.")
            }
            AclInnerError::AclInnerErrorProfilingFailure => write!(
                f,
                "Profiling related errors.\nYou can obtain the logs and contact technical support."
            ),
            // Runtime error messages
            AclInnerError::AclInnerErrorRtParamInvalid => write!(
                f,
                "Parameter validation failed.\nPlease check whether the interface input parameters are correct."
            ),
            AclInnerError::AclInnerErrorRtInvalidDeviceId => write!(
                f,
                "Invalid Device ID.\nPlease check whether the Device ID is legal."
            ),
            AclInnerError::AclInnerErrorRtContextNull => write!(
                f,
                "Context is empty.\nPlease check whether acl.rt.set_context or acl.rt.set_device is called."
            ),
            AclInnerError::AclInnerErrorRtStreamContext => write!(
                f,
                "The stream is not in the current Context.\nPlease check whether the Context where the Stream is located is consistent with the current Context."
            ),
            AclInnerError::AclInnerErrorRtModelContext => write!(
                f,
                "The model is not in the current Context.\nPlease check whether the loaded model is consistent with the current Context."
            ),
            AclInnerError::AclInnerErrorRtStreamModel => write!(
                f,
                "The stream is not in the current model.\nPlease check whether the Stream has been bound to the model."
            ),
            AclInnerError::AclInnerErrorRtEventTimestampInvalid => write!(
                f,
                "The event timestamp is invalid.\nPlease check whether the Event is created."
            ),
            AclInnerError::AclInnerErrorRtEventTimestampReversal => write!(
                f,
                "Event timestamps are reversed.\nPlease check whether the Event is created."
            ),
            AclInnerError::AclInnerErrorRtAddrUnaligned => write!(
                f,
                "Memory address is misaligned.\nPlease check whether the requested memory address is aligned."
            ),
            AclInnerError::AclInnerErrorRtFileOpen => write!(
                f,
                "Failed to open the file.\nPlease check if the file exists."
            ),
            AclInnerError::AclInnerErrorRtFileWrite => write!(
                f,
                "Failed to write file.\nPlease check whether the file exists or has write permission."
            ),
            AclInnerError::AclInnerErrorRtStreamSubscribe => write!(
                f,
                "The stream is not subscribed or is subscribed repeatedly.\nPlease check whether the current stream is subscribed or re-subscribed."
            ),
            AclInnerError::AclInnerErrorRtThreadSubscribe => write!(
                f,
                "The thread is not subscribed or is duplicated.\nPlease check whether the current thread is subscribed or resubscribed."
            ),
            AclInnerError::AclInnerErrorRtGroupNotSet => write!(f, "Group not set."),
            AclInnerError::AclInnerErrorRtGroupNotCreate => write!(
                f,
                "The corresponding Group is not created.\nPlease check whether the Group ID set when calling the interface is within the supported range. The value range of Group ID is: [0, (number of groups - 1)]."
            ),
            AclInnerError::AclInnerErrorRtStreamNoCbReg => write!(
                f,
                "The Stream corresponding to this callback is not registered with the thread.\nPlease check whether the stream has been registered to the thread and whether the acl.rt.subscribe_report interface has been called."
            ),
            AclInnerError::AclInnerErrorRtInvalidMemoryType => write!(
                f,
                "Invalid memory type.\nPlease check whether the memory type is legal."
            ),
            AclInnerError::AclInnerErrorRtInvalidHandle => write!(
                f,
                "Invalid resource handle.\nCheck whether the corresponding input and used parameters are correct."
            ),
            AclInnerError::AclInnerErrorRtInvalidMallocType => write!(
                f,
                "The type of memory requested is incorrect.\nCheck whether the corresponding input and the memory type used are correct."
            ),
            AclInnerError::AclInnerErrorRtWaitTimeout => write!(
                f,
                "The task execution timed out.\nPlease try to execute the task delivery interface again."
            ),
            AclInnerError::AclInnerErrorRtTaskTimeout => write!(
                f,
                "The task execution timed out.\nPlease check whether the service arrangement is reasonable or set a reasonable timeout period."
            ),
            AclInnerError::AclInnerErrorRtSysparamoptNotSet => write!(
                f,
                "Failed to obtain the system parameter value in the current Context.\nThe system parameter value in the current Context is not set."
            ),
            AclInnerError::AclInnerErrorRtFeatureNotSupport => write!(
                f,
                "Feature not supported.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtMemoryAllocation => write!(
                f,
                "Memory request failed.\nPlease check the remaining storage space on your hardware environment."
            ),
            AclInnerError::AclInnerErrorRtMemoryFree => write!(
                f,
                "Memory deallocation failed.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtAicoreOverFlow => write!(
                f,
                "The aicore operator overflowed.\nPlease check whether the corresponding aicore operator operation has overflow."
            ),
            AclInnerError::AclInnerErrorRtNoDevice => write!(
                f,
                "Device is unavailable.\nPlease check whether the device is running normally."
            ),
            AclInnerError::AclInnerErrorRtResourceAllocFail => write!(
                f,
                "Memory request failed.\nPlease check the remaining storage space on your hardware environment."
            ),
            AclInnerError::AclInnerErrorRtNoPermission => write!(
                f,
                "No operation permission.\nPlease check that the user permissions for running the application are correct."
            ),
            AclInnerError::AclInnerErrorRtNoEventResource => write!(
                f,
                "Insufficient Event resources.\nPlease refer to the description of the acl.rt.create_event interface to check whether the number of events meets the requirements."
            ),
            AclInnerError::AclInnerErrorRtNoStreamResource => write!(
                f,
                "Insufficient stream resources.\nPlease refer to the description of the acl.rt.create_stream interface to check whether the number of streams meets the requirements."
            ),
            AclInnerError::AclInnerErrorRtNoNotifyResource => write!(
                f,
                "The system's internal Notify resources are insufficient.\nThere are too many concurrent tasks for data preprocessing or too many resources are consumed during model inference. It is recommended that you try to reduce concurrent tasks or uninstall some models."
            ),
            AclInnerError::AclInnerErrorRtNoModelResource => write!(
                f,
                "Insufficient model resources.\nIt is recommended to uninstall some models."
            ),
            AclInnerError::AclInnerErrorRtNoCdqResource => write!(
                f,
                "Insufficient Runtime internal resources.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtOverLimit => write!(
                f,
                "The number of queues exceeds the upper limit.\nPlease destroy unnecessary queues before creating new ones."
            ),
            AclInnerError::AclInnerErrorRtQueueEmpty => write!(
                f,
                "The queue is empty.\nYou cannot get data from an empty queue. Please add data to the queue first and then get it."
            ),
            AclInnerError::AclInnerErrorRtQueueFull => write!(
                f,
                "The queue is full.\nYou cannot add data to a full queue. Please get data from the queue first before adding it."
            ),
            AclInnerError::AclInnerErrorRtRepeatedInit => write!(
                f,
                "The queue is initialized repeatedly.\nIt is recommended to initialize the queue once and not repeatedly."
            ),
            AclInnerError::AclInnerErrorRtDeviceOom => write!(
                f,
                "The device memory is exhausted.\nCheck the memory usage on the device and plan the memory usage properly according to the memory specifications on the device."
            ),
            AclInnerError::AclInnerErrorRtInternalError => write!(
                f,
                "An internal error occurred in the runtime module on the Host.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtTsError => write!(
                f,
                "An internal error occurred in the task scheduler module on the device."
            ),
            AclInnerError::AclInnerErrorRtStreamTaskFull => {
                write!(f, "The number of tasks on the stream is full.")
            }
            AclInnerError::AclInnerErrorRtStreamTaskEmpty => {
                write!(f, "The number of tasks on the stream is empty.")
            }
            AclInnerError::AclInnerErrorRtStreamNotComplete => {
                write!(f, "Not all tasks on the stream have been completed.")
            }
            AclInnerError::AclInnerErrorRtEndOfSequence => {
                write!(f, "The task execution on the AI CPU is completed.")
            }
            AclInnerError::AclInnerErrorRtEventNotComplete => {
                write!(f, "The event is not completed.")
            }
            AclInnerError::AclInnerErrorRtContextReleaseError => {
                write!(f, "Failed to release the context.")
            }
            AclInnerError::AclInnerErrorRtSocVersion => write!(
                f,
                "Failed to obtain soc version.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtTaskTypeNotSupport => write!(f, "Unsupported task type."),
            AclInnerError::AclInnerErrorRtLostHeartbeat => {
                write!(f, "The task scheduler lost its heartbeat.")
            }
            AclInnerError::AclInnerErrorRtModelExecute => write!(f, "Model execution failed."),
            AclInnerError::AclInnerErrorRtReportTimeout => write!(
                f,
                "Failed to obtain task scheduler messages.\nCheck whether the interface timeout setting is too short and increase the timeout appropriately. If the timeout error still occurs after increasing the timeout, check the log again.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtSysDma => write!(
                f,
                "System dma (Direct Memory Access) hardware execution error.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtAicoreTimeout => write!(f, "aicore execution timeout."),
            AclInnerError::AclInnerErrorRtAicoreException => {
                write!(f, "aicore execution exception.")
            }
            AclInnerError::AclInnerErrorRtAicoreTrapException => {
                write!(f, "aicore trap execution exception.")
            }
            AclInnerError::AclInnerErrorRtAicpuTimeout => write!(f, "aicpu execution timeout."),
            AclInnerError::AclInnerErrorRtAicpuException => write!(f, "aicpu execution exception."),
            AclInnerError::AclInnerErrorRtAicpuDatadumpRspErr => write!(
                f,
                "After aicpu executes data dump, it does not return a response to the task scheduler."
            ),
            AclInnerError::AclInnerErrorRtAicpuModelRspErr => write!(
                f,
                "After aicpu executes the model, it does not return a response to the task scheduler."
            ),
            AclInnerError::AclInnerErrorRtProfilingError => write!(
                f,
                "The profiling function is executed abnormally.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtIpcError => write!(
                f,
                "Inter-process communication exception.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtModelAbortNormal => write!(f, "Model exits."),
            AclInnerError::AclInnerErrorRtKernelUnregistering => {
                write!(f, "The operator is being registered.")
            }
            AclInnerError::AclInnerErrorRtRingbufferNotInit => {
                write!(f, "The ringbuffer function is not initialized.")
            }
            AclInnerError::AclInnerErrorRtRingbufferNoData => {
                write!(f, "The ringbuffer has no data.")
            }
            AclInnerError::AclInnerErrorRtKernelLookup => {
                write!(f, "The kernel inside RUNTIME is not registered.")
            }
            AclInnerError::AclInnerErrorRtKernelDuplicate => {
                write!(f, "Repeatedly register the kernel inside RUNTIME.")
            }
            AclInnerError::AclInnerErrorRtDebugRegisterFail => {
                write!(f, "Failed to register the debug function.")
            }
            AclInnerError::AclInnerErrorRtDebugUnregisterFail => {
                write!(f, "The debug function registration failed.")
            }
            AclInnerError::AclInnerErrorRtLabelContext => write!(
                f,
                "The tag is not in the current Context.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtProgramUseOut => {
                write!(f, "The number of registered programs exceeds the limit.")
            }
            AclInnerError::AclInnerErrorRtDevSetupError => write!(f, "Device startup failed."),
            AclInnerError::AclInnerErrorRtVectorCoreTimeout => {
                write!(f, "Vector core execution timeout.")
            }
            AclInnerError::AclInnerErrorRtVectorCoreException => {
                write!(f, "Vector core execution exception.")
            }
            AclInnerError::AclInnerErrorRtVectorCoreTrapException => {
                write!(f, "Vector core trap execution exception.")
            }
            AclInnerError::AclInnerErrorRtCdqBatchAbnormal => {
                write!(f, "Runtime internal resource request exception.")
            }
            AclInnerError::AclInnerErrorRtDieModeChangeError => write!(
                f,
                "The die mode modification is abnormal and the die mode cannot be modified."
            ),
            AclInnerError::AclInnerErrorRtDieSetError => {
                write!(f, "The die cannot be specified in single die mode.")
            }
            AclInnerError::AclInnerErrorRtInvalidDieid => write!(
                f,
                "The die id was incorrectly specified.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtDieModeNotSet => write!(f, "The die mode is not set."),
            AclInnerError::AclInnerErrorRtAicoreTrapReadOverflow => {
                write!(f, "aicore trap read out of bounds exception.")
            }
            AclInnerError::AclInnerErrorRtAicoreTrapWriteOverflow => {
                write!(f, "aicore trap write out of bounds exception.")
            }
            AclInnerError::AclInnerErrorRtVectorCoreTrapReadOverflow => {
                write!(f, "Vector core trap read out of bounds exception.")
            }
            AclInnerError::AclInnerErrorRtVectorCoreTrapWriteOverflow => {
                write!(f, "Vector core trap write out of bounds exception.")
            }
            AclInnerError::AclInnerErrorRtStreamSyncTimeout => write!(
                f,
                "In the specified timeout wait event, all tasks in the specified stream have not been completed."
            ),
            AclInnerError::AclInnerErrorRtEventSyncTimeout => write!(
                f,
                "In the specified Event synchronization waiting, the specified time has passed and the Event has not been completed."
            ),
            AclInnerError::AclInnerErrorRtFftsPlusTimeout => {
                write!(f, "Internal task execution timeout.")
            }
            AclInnerError::AclInnerErrorRtFftsPlusException => write!(
                f,
                "Internal task execution exception.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorRtFftsPlusTrapException => {
                write!(f, "Internal task trap exception.")
            }
            AclInnerError::AclInnerErrorRtSendMsg => {
                write!(f, "Message sending failed during data enqueuing.")
            }
            AclInnerError::AclInnerErrorRtCopyData => {
                write!(f, "Memory copy failed during data enqueue process.")
            }
            AclInnerError::AclInnerErrorRtDrvInternalError => {
                write!(f, "Internal error in the Driver module.")
            }
            AclInnerError::AclInnerErrorRtAicpuInternalError => {
                write!(f, "AI CPU module internal error.")
            }
            AclInnerError::AclInnerErrorRtSocketClose => write!(
                f,
                "The internal Host Device Communication (HDC) session link is disconnected."
            ),
            AclInnerError::AclInnerErrorRtAicpuInfoLoadRspErr => {
                write!(f, "AI CPU scheduling process failed.")
            }
            AclInnerError::AclInnerErrorGeParamInvalid => write!(
                f,
                "Parameter validation failed.\nPlease check whether the input parameter value of the interface is correct."
            ),
            AclInnerError::AclInnerErrorGeExecNotInit => write!(
                f,
                "Not initialized.\nPlease check whether the acl.init interface has been called for initialization. Please make sure that the acl.init interface has been called and that it is called before other pyACL interfaces.\nPlease check whether the initialization interface of the corresponding function has been called, such as the acl.mdl.init_dump interface for initializing Dump and the acl.prof.init interface for initializing Profiling."
            ),
            AclInnerError::AclInnerErrorGeExecModelPathInvalid => write!(
                f,
                "Invalid model path.\nPlease check if the model path is correct."
            ),
            AclInnerError::AclInnerErrorGeExecModelIdInvalid => write!(
                f,
                "Invalid model ID.\nPlease check whether the model ID is correct and whether the model is loaded correctly."
            ),
            AclInnerError::AclInnerErrorGeExecModelDataSizeInvalid => {
                write!(f, "Invalid model size.\nInvalid model size.")
            }
            AclInnerError::AclInnerErrorGeExecModelAddrInvalid => write!(
                f,
                "Invalid model memory address.\nPlease check that the model address is valid."
            ),
            AclInnerError::AclInnerErrorGeExecModelQueueIdInvalid => {
                write!(f, "Invalid queue ID.\nInvalid queue ID.")
            }
            AclInnerError::AclInnerErrorGeExecLoadModelRepeated => write!(
                f,
                "Repeat initialization or repeat loading.\nPlease check whether the corresponding interface is called to initialize or load repeatedly."
            ),
            AclInnerError::AclInnerErrorGeDynamicInputAddrInvalid => write!(
                f,
                "Invalid dynamic bin input address.\nPlease check the dynamic bin input address."
            ),
            AclInnerError::AclInnerErrorGeDynamicInputLengthInvalid => write!(
                f,
                "Invalid dynamic bin input length.\nPlease check the dynamic bin input length."
            ),
            AclInnerError::AclInnerErrorGeDynamicBatchSizeInvalid => write!(
                f,
                "Invalid dynamic batch size.\nPlease check the dynamic batch size."
            ),
            AclInnerError::AclInnerErrorGeAippBatchEmpty => write!(
                f,
                "Invalid AIPP batch number.\nPlease check whether the AIPP batch number is correct."
            ),
            AclInnerError::AclInnerErrorGeAippNotExist => write!(
                f,
                "The AIPP configuration does not exist.\nPlease check whether AIPP is configured."
            ),
            AclInnerError::AclInnerErrorGeAippModeInvalid => write!(
                f,
                "Invalid AIPP mode.\nPlease check whether the AIPP mode configured during model conversion is correct."
            ),
            AclInnerError::AclInnerErrorGeOpTaskTypeInvalid => write!(
                f,
                "Invalid task type.\nPlease check whether the operator type is correct."
            ),
            AclInnerError::AclInnerErrorGeOpKernelTypeInvalid => write!(
                f,
                "Invalid operator type.\nPlease check whether the operator type is correct."
            ),
            AclInnerError::AclInnerErrorGePlgmgrPathInvalid => write!(
                f,
                "Invalid so files, including those whose path levels are too deep or whose so files are deleted by mistake.\nPlease check whether the environment variable LD_LIBRARY_PATH configured before running the application is correct. For detailed description, please refer to the operation guide of the compilation and operation."
            ),
            AclInnerError::AclInnerErrorGeFormatInvalid => write!(
                f,
                "Invalid format.\nPlease check whether the format of the Tensor data is valid."
            ),
            AclInnerError::AclInnerErrorGeShapeInvalid => write!(
                f,
                "Invalid shape.\nPlease check whether the shape of the Tensor data is valid."
            ),
            AclInnerError::AclInnerErrorGeDatatypeInvalid => write!(
                f,
                "Invalid data type.\nPlease check whether the data type of the Tensor data is valid."
            ),
            AclInnerError::AclInnerErrorGeMemoryAllocation => write!(
                f,
                "Failed to apply for memory.\nPlease check the remaining memory on your hardware environment."
            ),
            AclInnerError::AclInnerErrorGeMemoryOperateFailed => write!(
                f,
                "Memory initialization or memory copy operation failed.\nPlease check whether the memory address is correct and whether there is enough memory in the hardware environment."
            ),
            AclInnerError::AclInnerErrorGeDeviceMemoryAllocationFailed => write!(
                f,
                "Failed to apply for Device memory.\nThe device memory is used up and the application cannot be continued. Please free up some device memory and try again."
            ),
            AclInnerError::AclInnerErrorGeUserRaiseException => write!(
                f,
                "User-defined functions actively throw exceptions.\nUsers can identify which input data caused an error based on the UserData set in DataFlowInfo, and then troubleshoot the problem based on the error."
            ),
            AclInnerError::AclInnerErrorGeDataNotAligned => write!(
                f,
                "Data is misaligned.\nIf a user-defined function has multiple outputs, you need to check whether there are fewer outputs set in the user code. Missing outputs may cause data alignment exceptions."
            ),
            AclInnerError::AclInnerErrorGeInternalError => write!(
                f,
                "Unknown internal error.\nYou can obtain the logs and contact technical support."
            ),
            AclInnerError::AclInnerErrorGeLoadModel => {
                write!(f, "The system failed to load the model internally.")
            }
            AclInnerError::AclInnerErrorGeExecLoadModelPartitionFailed => {
                write!(f, "The system failed to load the model internally.")
            }
            AclInnerError::AclInnerErrorGeExecLoadWeightPartitionFailed => {
                write!(f, "The system failed to load the model weights internally.")
            }
            AclInnerError::AclInnerErrorGeExecLoadTaskPartitionFailed => {
                write!(f, "The system internal model loading task failed.")
            }
            AclInnerError::AclInnerErrorGeExecLoadKernelPartitionFailed => {
                write!(f, "The system failed to load the model operator.")
            }
            AclInnerError::AclInnerErrorGeExecReleaseModelData => {
                write!(f, "Failed to release model space in the system.")
            }
            AclInnerError::AclInnerErrorGeCommandHandle => {
                write!(f, "The command operation within the system failed.")
            }
            AclInnerError::AclInnerErrorGeGetTensorInfo => {
                write!(f, "Failed to obtain tensor data in the system.")
            }
            AclInnerError::AclInnerErrorGeUnloadModel => {
                write!(f, "Failed to unload the model space in the system.")
            }
            AclInnerError::AclInnerErrorGeModelExecuteTimeout => {
                write!(f, "Model execution timed out.")
            }
        }
    }
}

impl std::error::Error for AclInnerError {}

impl From<i32> for AclInnerError {
    fn from(value: i32) -> Self {
        AclInnerError::from_i32(value).unwrap_or(AclInnerError::AclInnerErrorFailure)
    }
}

impl From<AclInnerError> for i32 {
    fn from(error: AclInnerError) -> Self {
        error as i32
    }
}
