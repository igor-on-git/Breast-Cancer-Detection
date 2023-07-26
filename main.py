import metrics
from utils import *
from train import *
from test import *
from model_selector import *

if __name__ == '__main__':

    ## parameters
    model_name_list = ['densenet121'] #['resnet152', 'densenet121', 'IgorNetV1', 'IgorNetV2',]
    train_en = 1
    continue_training = 0
    train_epochs = 2

    #batch_size = 128
    #learn_rate = 0.001 / 2
    #train_stop_criteria = 'valid loss' # 'F1 score'
    #train_stop_patience = 5

    ## one time prep data to match imagefolder functionality
    # https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
    source_data_folder = './data/'
    dest_data_folder = './data_reordered/'
    data_split = [0.8, 0.1, 0.1]
    if os.path.exists(dest_data_folder) == 0:
        train_files, valid_files, test_files = load_data(source_data_folder, data_split)
        reorder_data_for_image_folder(train_files, dest_data_folder + 'train/')
        reorder_data_for_image_folder(valid_files, dest_data_folder + 'valid/')
        reorder_data_for_image_folder(test_files, dest_data_folder + 'test/')

    ## generate dataloaders
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize(50),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(50),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(dest_data_folder + 'train/', transform=train_transforms)
    valid_data = datasets.ImageFolder(dest_data_folder + 'valid/', transform=test_transforms)
    test_data = datasets.ImageFolder(dest_data_folder + 'train/', transform=test_transforms)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #test_loss_all = np.ones(len(model_name_list))
    #precision_all = np.ones(len(model_name_list))
    #recall_all = np.ones(len(model_name_list))
    #F1score_all = np.ones(len(model_name_list))
    test_perf_all = [dict() for x in range(len(model_name_list))]

    for model_ind, model_name in enumerate(model_name_list):

        model, optimizer, criterion, batch_size, learn_rate, train_stop_criteria, train_stop_patience = model_selector(model_name)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        if train_en:

            if continue_training:
                model.load_state_dict(torch.load('models/' + model_name + '/state_dict.pth'))
                train_perf = np.load('models/' + model_name + '/train_perf.npy', allow_pickle=True).item()
            else:
                train_perf = None

            model, train_perf = train_model(
                model_name, model, criterion, optimizer, trainloader, validloader, device, train_epochs, train_stop_criteria, train_stop_patience, perf_prev=train_perf)

        else:

            model.load_state_dict(torch.load('models/' + model_name + '/state_dict.pth'))
            train_perf = {}

        #test_loss_all[model_ind], precision_all[model_ind], recall_all[model_ind], F1score_all[model_ind] = \
        test_perf_all[model_ind] = test(model_name, model, criterion, testloader, device)
        np.save('models/' + model_name + '/test_perf.npy', test_perf_all[model_ind])
                #{'test_loss': test_loss_all[model_ind], 'precision': precision_all[model_ind], 'recall': recall_all[model_ind], 'F1score': F1score_all[model_ind]})

    test_metric_all = [train_perf[model_name_list]['F1score'] for i in range(len(model_name_list))]
    best_ind = np.argwhere(test_metric_all == np.max(test_metric_all))
    print('Best(F1 score) Net is: {}'.format(model_name_list[int(best_ind)]))