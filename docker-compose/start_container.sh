docker run -d \
--name linux \
--mount source=/Users/moyanxinxu/Documents/unlock-hf,target=/root/Documents/unlock-hf,type=bind \
-v huggingface:/root/.cache/huggingface/ \
-it moyanxinxu/self-ubuntu-myxx:v3.0 /bin/bash
