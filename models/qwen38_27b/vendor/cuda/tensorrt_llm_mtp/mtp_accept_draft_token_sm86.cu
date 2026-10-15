/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// ref: upstream/mtpKernels.cu:250-312
extern "C" __global__ void ctox_trtllm_mtp_accept_draft_token_sm86(int const numMTPModules,
    int const batchSize, int const numContextRequest, int const* draftTokens, int* targetTokens,
    int* acceptedTokens, int* numAcceptedTokens)
{
    /*
        In a batch of request: context request (at the beginning) + generation requests
        numGenerationRequest = batchSize - numContextRequest
        numLogits = numContextRequest + numGenerationRequest * (numMTPModules + 1)
        allDraftToken = numGenerationRequest * numMTPModules

        draftTokens: [allDraftToken], flatten, remove padding
        targetTokens: [numLogits], temporary buffer
        acceptedTokens: [batchSize, numMTPModules + 1]
        numAcceptedTokens: [batchSize]
    */
    int const bid = static_cast<int>(blockIdx.x);
    int const tid = static_cast<int>(bid * blockDim.x + threadIdx.x);

    if (tid < batchSize)
    {
        // For the context requests, curDraftLen == 0
        // For the generation requests, curDraftLen == numMTPModules
        int curDraftLen = 0;
        if (tid >= numContextRequest)
        {
            // Generation request
            curDraftLen = numMTPModules;
        }

        int draftTokensStartOffset = -1;
        int targetTokensStartOffset = -1;

        if (tid < numContextRequest)
        {
            // Context requests
            draftTokensStartOffset = 0;    // context requests do not have draft tokens
            targetTokensStartOffset = tid; // the associated logits index
        }
        else
        {
            // Generation requests
            draftTokensStartOffset = (tid - numContextRequest) * numMTPModules;
            targetTokensStartOffset = numContextRequest + (tid - numContextRequest) * (numMTPModules + 1);
        }

        // Compare the draft tokens and target tokens
        int curAcceptedLen = 0;
        while ((curAcceptedLen < curDraftLen)
            && (draftTokens[draftTokensStartOffset + curAcceptedLen]
                == targetTokens[targetTokensStartOffset + curAcceptedLen]))
        {
            curAcceptedLen++;
        }
        curAcceptedLen++; // one more for the golden token
        numAcceptedTokens[tid] = curAcceptedLen;

        // Write back to acceptedTokens
        auto curAcceptedTokensPtr = acceptedTokens + tid * (numMTPModules + 1);
        for (int jj = 0; jj < curAcceptedLen; jj++)
        {
            curAcceptedTokensPtr[jj] = targetTokens[targetTokensStartOffset + jj];
        }
    }
}
