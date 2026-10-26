# Check observed setup failure and cleanup without treating null samples as data.
def require($condition; $message):
  if $condition then . else error($message) end;
require(length == 14; "incomplete or unexpected records")
| require(.[0] == {
    kind:"preflight", device_index:0, effective_uid:1000,
    compiled_cupti_api:26, sampling_period:"MID", sampling_period2:0
  }; "wrong observation configuration")
| require([.[] | select(.kind == "api") | .api] == [
    "cuInit", "cuDriverGetVersion", "cuDeviceGet", "cuDeviceGetName",
    "cuDeviceGetAttribute.major", "cuDeviceGetAttribute.minor", "cuCtxCreate",
    "cuptiGetVersion", "cuptiActivityConfigurePCSampling", "cuCtxDestroy"
  ]; "API sequence or cleanup missing")
| require(all(.[] | select(.kind == "api");
    if .api == "cuptiActivityConfigurePCSampling" then
      .status == 27 and .message == "CUPTI_ERROR_NOT_SUPPORTED"
    else .status == 0 end); "wrong failure or failed cleanup")
| require([.[] | select(.kind == "device")] == [{
    kind:"device", name:"Tesla M40 24GB", major:5, minor:2,
    driver_api_version:13000
  }]; "wrong device or driver API")
| require([.[] | select(.kind == "sdk")] == [{kind:"sdk",runtime_cupti_api:26}];
    "wrong runtime SDK API")
| require(.[-1] == {
    kind:"result", exit_code:2, has_setup_support:false,
    has_clean_context:true, has_kernel_measurements:false, kernel_launches:0,
    samples:null
  }; "false availability/measurement claim or missing final result")
| {
    passes_audit:true,
    scope:"untimed setup capability, no kernels or measured samples",
    configure_status:27,
    configure_message:"CUPTI_ERROR_NOT_SUPPORTED",
    cleanup_status:0,
    exit_code:2,
    has_kernel_measurements:false,
    samples:null,
    changes_runtime_defaults:false
  }
