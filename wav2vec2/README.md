## https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h


https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py


https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h/blob/main/pytorch_model.bin


```py
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
```


- - -


https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

```py
class Net(nn.Module):
        // Your Model for which you want to load parameters 

model = Net()
torch.optim.SGD(lr=0.001) #According to your own Configuration.
checkpoint = torch.load(pytorch_model)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['opt']) 
```
