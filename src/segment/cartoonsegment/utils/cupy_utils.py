import re
import os
import cupy
import os.path as osp
import torch

@cupy.memoize(for_each_device=True)
def launch_kernel(strFunction, strKernel):
    if 'CUDA_HOME' not in os.environ:
        os.environ['CUDA_HOME'] = cupy.cuda.get_cuda_path()
    # end
    # , options=tuple([ '-I ' + os.environ['CUDA_HOME'], '-I ' + os.environ['CUDA_HOME'] + '/include' ])
    return cupy.RawKernel(strKernel, strFunction)
    

def preprocess_kernel(strKernel, objVariables):
    path_to_math_helper = osp.join(osp.dirname(osp.abspath(__file__)), 'helper_math.h')
    strKernel = '''
        #include <{{HELPER_PATH}}>

        __device__ __forceinline__ float atomicMin(const float* buffer, float dblValue) {
            int intValue = __float_as_int(*buffer);

            while (__int_as_float(intValue) > dblValue) {
                intValue = atomicCAS((int*) (buffer), intValue, __float_as_int(dblValue));
            }

            return __int_as_float(intValue);
        }


        __device__ __forceinline__ float atomicMax(const float* buffer, float dblValue) {
            int intValue = __float_as_int(*buffer);

            while (__int_as_float(intValue) < dblValue) {
                intValue = atomicCAS((int*) (buffer), intValue, __float_as_int(dblValue));
            }

            return __int_as_float(intValue);
        }
    '''.replace('{{HELPER_PATH}}', path_to_math_helper) + strKernel
    # end

    for strVariable in objVariables:
        objValue = objVariables[strVariable]

        if type(objValue) == int:
            strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

        elif type(objValue) == float:
            strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

        elif type(objValue) == str:
            strKernel = strKernel.replace('{{' + strVariable + '}}', objValue)

        # end
    # end

    while True:
        objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg] if torch.is_tensor(intSizes[intArg]) == False else intSizes[intArg].item()))
    # end

    while True:
        objMatch = re.search('(STRIDE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intStrides = objVariables[strTensor].stride()

        strKernel = strKernel.replace(objMatch.group(), str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()))
    # end

    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')' for intArg in range(intArgs) ]

        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
    # end

    while True:
        objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')' for intArg in range(intArgs) ]

        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end
    return strKernel