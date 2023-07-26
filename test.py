import torch
import metrics


def test(model_name, model, criterion, test_loader, device):

    # initialize lists to monitor test loss and accuracy
    test_perf = init_test_perf()
    ref_vals = []
    est_vals = []
    model.to(device)
    model.eval()  # prep model for evaluation

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update test loss
            test_perf['test_loss'] += loss.item() * data.size(0)
            # convert output probabilities to predicted class
            ps = torch.exp(output)
            top_prob, top_class = ps.topk(1, dim=1)
            ref_vals.extend(target.cpu().numpy().reshape(-1))
            est_vals.extend(top_class.cpu().numpy().reshape(-1))

        # calculate and print avg test loss
        test_perf['test_loss'] /= len(test_loader.dataset)
        test_perf['precision'] = metrics.precision(ref_vals, est_vals)
        test_perf['recall'] = metrics.recall(ref_vals, est_vals)
        test_perf['F1score'] = metrics.F1score(ref_vals, est_vals)
        print('Model: {}\tTest Loss: {:.3f}\tPrecision: {:.3f}%\tRecall: {:.3f}%\tF1-score: {:.3f}\n'.format(model_name, test_perf['test_loss'], test_perf['precision']*100, test_perf['recall']*100, test_perf['F1score']))

    return test_perf


def init_test_perf():

    test_perf = {}
    test_perf['test_loss'] = 0
    test_perf['precision'] = 0
    test_perf['recall'] = 0
    test_perf['F1score'] = 0

    return test_perf