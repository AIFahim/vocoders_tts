import torch

# checkpoint = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/best_model_296927.pth")
checkpoint = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/univnet_weights/48_sr_model_weight/checkpoint_160000.pth")
# print(checkpoint.keys())

# Extract the state dictionary
state_dict = checkpoint['model']

# for key in state_dict.keys():
#     print(key)

# Modify the state dictionary
new_state_dict = {}
for key, value in state_dict.items():
    if not key.startswith('model_d.'):
        new_key = key.replace('model_g.', '')
        new_state_dict[new_key] = value

# Update the 'model' key in the checkpoint with the modified state dictionary
checkpoint['model'] = new_state_dict

# Save the modified checkpoint
torch.save(checkpoint, '/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/univnet_weights/renamed_weights_44k_160000.pth')

checkpoint_new = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/univnet_weights/renamed_weights_44k_160000.pth")
# Print the new keys in the checkpoint
print(checkpoint_new.keys())

for key in checkpoint_new['model'].keys():
    print(key)

# import torch

# checkpoint = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/best_model_296927.pth")
# print(checkpoint.keys())

# # Extract the state dictionary
# state_dict = checkpoint['model']


# for key in state_dict.keys():
#     print(key)


# # Modify the state dictionary
# new_state_dict = {}
# for key, value in state_dict.items():
#     if key.startswith('model_g.'):
#         new_key = key.replace('model_g.', '')
#     elif key.startswith('model_d.discriminators.0.'):
#         new_key = key.replace('model_d.discriminators.0.', '')
#     elif key.startswith('model_d.discriminators.1.'):
#         new_key = key.replace('model_d.discriminators.1.', '')
#     elif key.startswith('model_d.discriminators.2.'):
#         new_key = key.replace('model_d.discriminators.2.', '')
#     # elif key.startswith('discriminators.'):
#     #     new_key = key.replace('discriminators', '')

#     else:
#         new_key = key
#     new_state_dict[new_key] = value

# # Update the 'model' key in the checkpoint with the modified state dictionary
# checkpoint['model'] = new_state_dict

# # Save the modified checkpoint
# torch.save(checkpoint, '/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/renamed_weights.pth')

# checkpoint_new = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/renamed_weights.pth")
# # Print the new keys in the checkpoint
# print(checkpoint_new.keys())

# for key in checkpoint_new['model'].keys():
#     print(key)





# import torch

# checkpoint = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/best_model_296927.pth")
# print(checkpoint.keys())

# # Extract the state dictionary
# state_dict = checkpoint['model']

# # Modify the state dictionary
# new_state_dict = {k.replace('model_g.', ''): v for k, v in state_dict.items()}
# new_state_dict = {k.replace('model_d.', ''): v for k, v in state_dict.items()}

# # Update the 'model' key in the checkpoint with the modified state dictionary
# checkpoint['model'] = new_state_dict

# # Save the modified checkpoint
# torch.save(checkpoint, '/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/renamed_weights.pth')

# checkpoint_new = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/renamed_weights.pth")
# # Print the new keys in the checkpoint
# print(checkpoint_new.keys())

# for key in checkpoint_new['model'].keys():
#     print(key)





# import torch

# checkpoint = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/best_model_296927.pth")
# print(checkpoint.keys())

# # Extract the state dictionary
# state_dict = checkpoint['model']

# # Modify the state dictionary
# new_state_dict = {k.replace('model_g.', ''): v for k, v in state_dict.items()}
# new_state_dict = {k.replace('model_d.', ''): v for k, v in new_state_dict.items()}

# # Save the modified state dictionary
# torch.save(new_state_dict, '/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/renamed_weights.pth')


# checkpoint_new = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/renamed_weights.pth")
# # Print the new keys in the checkpoint
# print(checkpoint_new.keys())




# import torch
# checkpoint = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/best_model_296927.pth")
# state_dict = checkpoint['model']
# new_state_dict = {k.replace('model_g.', ''): v for k, v in state_dict.items()}
# new_state_dict = {k.replace('model_d.', ''): v for k, v in state_dict.items()}

# for key in new_state_dict.keys():
#     print(key)
# # model.load_state_dict(new_state_dict)
# # torch.save(model.state_dict(), 'corrected_model.pth')
# torch.save(new_state_dict, '/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/renamed_weights.pth')

# import torch

# # Load the checkpoint
# checkpoint = torch.load("/home/asif/tts_all/coqui_tts/my_exp/coqui_vocoder/multibandmelgan_weights/best_model_296927.pth")

# # Extract the state dictionary
# state_dict = checkpoint['model']

# # Print the keys (layer names)
# for key in state_dict.keys():
#     print(key)
