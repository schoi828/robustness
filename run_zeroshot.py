import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, InstructBlipProcessor, InstructBlipForConditionalGeneration,  AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM 
import torch
from domainbed.iwildcam import iwildcam
from domainbed.fmow import FMoW
from domainbed.camelyon17 import Camelyon17 
import numpy as np
from PIL import Image
import argparse
from domainbed import datasets
import sys
from domainbed.lib import misc
import time
import transformers
import re
import base64
import asyncio
import nest_asyncio
nest_asyncio.apply()
from io import BytesIO

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def pil_to_base64(pil_img):
  img_buffer = BytesIO()
  pil_img.save(img_buffer, format='JPEG')
  byte_data = img_buffer.getvalue()
  base64_str = base64.b64encode(byte_data).decode('utf-8')
  return base64_str
    
class Zeroshot:
    def __init__(self, args):

        self.mode = args.mode
        self.prompt_type = args.prompt_type
        self.dataset = args.dataset
        self.api_key = args.api_key
        self.root = args.root
        if args.dataset == 'iwildcam':
            self.data = iwildcam(root=self.root, split='test',algo='zeroshot')
        elif args.dataset == 'FMOW':
            self.data = FMoW(root=self.root, split='test',algo='zeroshot')
        elif args.dataset == 'CAMELYON17':
            self.data = Camelyon17(root=self.root, split='test',algo='zeroshot')
        else:
            self.data = vars(datasets)[args.dataset](root=self.root, dist_type='NONE', dataset_size=1, split='test')

        self.labels = self.data.label_names

        #A list of labels will be included in the input text prompt
        label_list = ''
        for l_i, label in enumerate(self.labels):
            label = label.replace("_", " ")
            if l_i == len(self.labels)-1:
                label_list+=' or '
            else:
                label_list+=' '
            label_list+=f'\'{label}\''
        self.label_list = label_list
        self.get_model()
        self.get_prompt()

    def get_prompt(self):
        tailored_dict = {
                        "DSPRITES": f"Classify the object in the image into {self.label_list}. Please provide only the name of the label.", 
                        "SMALLNORB": f"Classify the object in the image into {self.label_list}. Please provide only the name of the label.", 
                        "SHAPES3D": f"Classify the object in the image into {self.label_list}. Please provide only the name of the label.", 
                        "DEEPFASHION": "Is a person wearing a dress or not? Please answer in yes or no.",
                        "CELEBA": f"Classify the person in the image into {self.label_list}. Please provide only the name of the label.",
                        "iwildcam": f"Classify the object or animal in the image. Here is the list of labels to choose from: {self.label_list}. Please provide only the name of the label.",
                        "FMOW": f"Classify the building or land-use in the image into {self.label_list}. Please provide only the name of the label."
                        }
        prompt_dict = {
                        'general'  :  f"Classify the image into {self.label_list}. Please provide only the name of the label.",
                        'general2' :  f"Choose a label that best describes the image. Here is the list of labels to choose from: {self.label_list}. Please provide only the name of the label.",
                        'tailored' : tailored_dict[self.dataset]
                        }       
        if 'CLIP' in self.mode:
            prompt_dict = {'general':[f"a photo of a {label.replace('_', ' ')}" for label in self.labels]}
        else:
            if self.dataset == 'DEEPFASHION' and self.prompt_type == 'tailored':
                self.labels = ['yes', 'no']
            elif self.dataset == 'iwildcam' and self.mode == 'InstructBLIP':
                #with iwildcam labels, the prompt's tokens exceed InstructBLIP's context window. 
                prompt_dict = {
                    'general'  :  "Classify the image. Please provide only the name of the label.",
                    'general2' :  "Choose a label that best describes the image. Please provide only the name of the label.",
                    'tailored'  :  "Classify the object or animal in the image. Please provide only the name of the label."
                    }

            elif self.dataset == 'CAMELYON17':
                prompt_dict = {
                'general'  :  "Please answer yes if the image contains any tumor tissue, and no otherwise. Please provide only the name of the label.",
                'general2'  :  "Please answer yes if the image contains any tumor tissue, and no otherwise. Please respond with a single word.",
                'tailored'  :  "Please analyze the image and determine if it contains any tumor tissue. Respond with 'Yes' if tumor tissue is present, or 'No' if it is not.",
                            }
                self.label_list = ['yes', 'no']                       
            
        self.prompt = self.format_prompt(prompt_dict[self.prompt_type])

        print(self.prompt)

    def format_prompt(self, prompt):
        if 'Phi' in self.mode:
            prompt= "USER: <|image_1|>\n"+prompt
        elif 'gpt' in self.mode:
            prompt = {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt,
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url":  ''
                        },
                        },
                    ],
                    }
        elif self.mode in ['InstructBLIP','LLAVA']:
            prompt = "USER: <image>\n"+ prompt + "\nASSISTANT:"

        return prompt

    def get_model(self):

        if self.mode == 'InstructBLIP':
            model_id = "Salesforce/instructblip-vicuna-7b"
            self.processor = InstructBlipProcessor.from_pretrained(model_id)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id,load_in_4bit=True, torch_dtype=torch.float16)

        elif self.mode == 'LLAVA':

            model_id = "llava-hf/llava-1.5-7b-hf"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
            ).to(0)

            self.processor = AutoProcessor.from_pretrained(model_id, revision='a272c74')
        
        elif 'Phi' in self.mode:
            model_id = "microsoft/Phi-3.5-vision-instruct"
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

        elif 'gpt' in self.mode:
            from openai import AsyncOpenAI            
            self.client = AsyncOpenAI(api_key=self.api_key)

        elif 'CLIP' in self.mode:

            id_dict = {'CLIP-base':"openai/clip-vit-base-patch32", 'CLIP-large':"openai/clip-vit-large-patch14"}
            model_id = id_dict[self.mode]
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to("cuda")
        
        else:
            raise NotImplementedError("The model is not available")
        
    def get_pred(self, text):
        try:
            pred = self.labels.index(text)
        except:
            pred = -1
            if self.dataset == 'CAMELYON17':
                if 'yes' in text.lower():
                    pred = 1
                elif 'no' in text.lower():
                    pred = 0
                else:
                    print(f'Warning: unknown gt_answer: {text}')
            elif self.dataset == 'DEEPFASHION':
                if 'no' in text.lower() and 'dress' in text.lower():
                    pred = 1
                elif 'dress' in text.lower() and not 'no' in text.lower():
                    pred = 0
            else:
                for i, label in enumerate(self.labels):
                    label = re.sub(r'[^A-Za-z0-9 ]+', ' ', label)
                    label = label.strip().lower()
                    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
                    text = text.strip().lower()
                    if text in label:
                        pred = i
                    if label in text:
                        pred = i
                        break
        return pred

    def forward(self, im, gts=None):
        generated_text = 'not generated yet'

        if self.mode == 'InstructBLIP':
            inputs = self.processor(images=im, text=[self.prompt]*self.batch_size, return_tensors="pt").to(device="cuda", dtype=torch.float16)
            # autoregressively generate an answer
            outputs = self.model.generate(
                    **inputs,
                    num_beams=5,
                    max_new_tokens=20,
                    min_length=1,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    temperature=1
            )
            outputs[outputs == 0] = 2 # this line can be removed once https://github.com/huggingface/transformers/pull/24492 is fixed
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
            generated_text = generated_text[0].strip()
            pred = self.get_pred(generated_text)
        
        elif 'Phi' in self.mode:
            messages = [ 
            {"role": "user", "content": self.prompt}
            ] 
            prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(prompt, [im], return_tensors="pt").to("cuda:0") 

            generation_args = { 
                "max_new_tokens": 200, #if 'max_new_tokens' not in self.config else self.config['max_new_tokens'],#500, 
                "temperature": 0.2 # 0.0 if 'temperature' not in self.config else self.config['temperature'], 
            } 
            generation_args['do_sample'] = True if generation_args['temperature'] > 0 else False 
            generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

            # remove input tokens 
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            generated_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
            pred = self.get_pred(generated_text)

        elif self.mode == 'LLAVA':
            inputs = self.processor(self.prompt, im, return_tensors='pt').to(0, torch.float16)
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generated_text = self.processor.decode(output[0][2:], skip_special_tokens=True).split('ASSISTANT: ')[-1].lower()
            
            pred = self.get_pred(generated_text)
        
        elif 'CLIP' in self.mode:

            label_tokens = self.processor(
            text=self.prompt,
            padding=True,
            images=None,
            return_tensors='pt'
            ).to("cuda")
            # encode tokens to sentence embeddings
            label_emb = self.model.get_text_features(**label_tokens)
            # detach from pytorch gradient computation
            label_emb = label_emb.detach().cpu().numpy()

            label_emb = label_emb / np.linalg.norm(label_emb, axis=1,keepdims=True)

            image = self.processor(
            text=None,
            images=im,
            return_tensors='pt'
            )['pixel_values'].cuda()
            img_emb = self.model.get_image_features(image)
            img_emb = img_emb.detach().cpu().numpy()
            img_emb = img_emb / np.linalg.norm(img_emb, axis=1,keepdims=True)
            scores = np.dot(img_emb, label_emb.T)
            pred = np.argmax(scores)
            generated_text = self.prompt[pred]

        return pred, generated_text
    
    async def gpt_infer(self, messages):
        response = await self.client.chat.completions.create(
        model=self.mode,messages=messages)
        return response
        
    async def gather(self, async_list):
        return await asyncio.gather(*async_list)
    
    def gpt_inference(self, output_dir=None):

        pred = 'empty'
        correct=0
        async_size = 50
        async_list = []
        gt_list = []
        r = 0
        resume = 0

        #load progress from a log txt file if the inference was interrupted
        if os.path.exists(os.path.join(output_dir, 'progress.txt')):
            with open(os.path.join(output_dir, 'progress.txt'), 'r') as f:
                for line in f.readlines():
                    try:
                        _, output, gt = line.split('_')
                    except:
                        split_list = line.split('_')
                        gt = split_list[-1]
                        output = " ".join(split_list[1:-1])
                    pred = self.get_pred(output)
                    try:
                        correct+= (int(gt)==pred)
                    except:
                        print(output, pred, gt)
                    resume+=1
            print('resume acc:', correct/resume)
        

        for i in range(len(self.data)):
            if i < resume:
                continue
            if self.dataset in ['FMOW', 'iwildcam', 'CAMELYON17']:
                im, gt = self.data[i]
                gt = int(gt)
            else:
                im = Image.fromarray(np.uint8(self.data._imgs[i]*255))
                gt = int(self.data._labels[i])
            
            prompt = self.prompt
            prompt["content"][1]["image_url"]["url"]= f"data:image/jpeg;base64,{pil_to_base64(im)}"
            async_list.append(self.gpt_infer([prompt]))
            gt_list.append(gt)
            if (i % async_size == 0 and i >0) or i == len(self.data)-1:
                responses = asyncio.run(self.gather(async_list))
                for response, gt in zip(responses, gt_list):
                    pred = self.get_pred(response.choices[0].message.content)
                    correct+= (int(gt)==pred)
                    if output_dir is not None:
                        with open(os.path.join(output_dir, 'progress.txt'), 'a') as f:
                            f.write(f'{r}_{response.choices[0].message.content}_{gt}\n')
                    r+=1
                gt_list = []
                async_list = []
                print(f'{i+1}/{len(self.data)}| accuracy: {correct/(i+1):.4f}')
            else:
                continue
        
        return {'total': correct/len(self.data)}


    def inference(self):
        
        pred = 'empty'
        #pbar = tqdm(range(len(self.data)))
        correct_0 = 0
        correct_1 = 0
        correct=0

        for i in range(len(self.data)):
            # if self.dataset == 'iwildcam':
            #     im_path, gt = self.data.samples[i]
            #     im = Image.open(im_path)
            if self.dataset in ['FMOW', 'iwildcam', 'CAMELYON17']:
                im, gt = self.data[i]
                gt = int(gt)
            else:
                im = Image.fromarray(np.uint8(self.data._imgs[i]*255))
                gt = int(self.data._labels[i])
            
            #im = Image.fromarray(np.uint8(data._imgs[i]*255))
            pred, text = self.forward(im)
 
            if pred == -1:
                print(f'wrong output from an image with label {gt}: ', text)
            if self.dataset == 'iwildcam':
                if i < 492:
                    correct_0+=(int(gt)==pred)
                else:
                    correct_1+=(int(gt)==pred)
                correct=correct_0+correct_1
            else:
                correct+= (int(gt)==pred)
            #pbar.set_postfix(pred=pred)
            if i % 50 == 0:
                print(f'{i+1}/{len(self.data)}| pred:gt = {pred}:{gt}| {text}')
            #pbar.set_description(f'{generated_text}, {pred}')
        
        return {'total': correct/len(self.data)}

if __name__ == "__main__":
    start_time = time.time()

    transformers.logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description='Zeroshot classification with vision lanaguage models')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--mode', type=str, default="LLAVA", choices=['InstructBLIP', 'CLIP-base', 'CLIP-large', 'LLAVA', 'Phi-3.5', 'gpt-4o', 'gpt-4o-mini'], help='Beware. Running gpt models is very expensive')
    parser.add_argument('--prompt_type', type=str, default="general", choices=['general', 'general2','tailored'])
    parser.add_argument('--dataset', type=str, default="iwildcam", choices=['iwildcam','DSPRITES','SHAPES3D','SMALLNORB', 'DEEPFASHION','CELEBA', 'FMOW', 'CAMELYON17'])
    parser.add_argument('--output_dir', type=str, default="../rob_exps/zeroshot")
    parser.add_argument('--api_key', type=str, default="", help="API key is necessary for GPT inference")
    parser.add_argument('--root', type=str, default="/data", help="root directory path where datasets are located")

    args = parser.parse_args()
    args.output_dir = f"{args.output_dir}/{args.dataset}/{args.mode}-{args.prompt_type}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    exp = Zeroshot(args)
    if 'gpt' in args.mode:
        accuracies = exp.gpt_inference(args.output_dir)
    else:
        accuracies = exp.inference()
    total_time = time.time()-start_time

    for key in accuracies:
        acc = accuracies[key]
        with open(os.path.join(args.output_dir, f'done_testacc_{key}_{acc:.4f}'), 'w') as f:
                f.write('done\n')
                f.write(f"test accuracy: {acc:.5f}\n")
                f.write(f'Elapsed Time: {(total_time)/3600: .1f} hour\n')
    print('done' ,args.output_dir)