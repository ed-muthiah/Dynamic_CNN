import os
import time
import torch
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.CrossEntropyLoss()

def train(args, model, optimizer, dataloaders):
    trainloader, testloader = dataloaders

    best_testing_accuracy = 0.0
    train_loss=[]
    epoch_x=[]
    test_accuracies=[]
    # training
    for epoch in range(args.epochs):
        model.train()

        batch_time = time.time(); iter_time = time.time()
        for i, data in enumerate(trainloader):

            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)

            cls_scores = model(imgs, with_dyn=args.with_dyn)
            loss = criterion(cls_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0 and i != 0:
                print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                    time.time()-iter_time, loss.item()))
                iter_time = time.time()
        batch_time = time.time() - batch_time
        print('[epoch {} | time:{:.2f} | loss:{:.5f}]'.format(epoch, batch_time, loss.item()))
        print('-------------------------------------------------')
        train_loss.append(loss.item())
        epoch_x.append(epoch)

        if epoch % 1 == 0:
            testing_accuracy = evaluate(args, model, testloader)
            test_accuracies.append(testing_accuracy.item())
            print('testing accuracy: {:.3f}'.format(testing_accuracy))

            if testing_accuracy > best_testing_accuracy:
                ### compare the previous best testing accuracy and the new testing accuracy
                ### save the model and the optimizer --------------------------------
                #
                #
                print('Saving....')
                state = {
                    'state_dict': model.state_dict(),
                    'acc': testing_accuracy,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,}
                torch.save(state,'df_0_checkpoint.pth')
                best_testing_accuracy=testing_accuracy
                #
                #
                ### -----------------------------------------------------------------
                print('new best model saved at epoch: {}'.format(epoch))
    print('-------------------------------------------------')
    print('best testing accuracy achieved: {:.3f}'.format(best_testing_accuracy))
    #fig = plt.plot(epoch_x, train_loss)
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(epoch_x, train_loss)
    axs[0, 0].set_title('Epoch vs. Training Loss')
    #axs[0, 1].plot(epoch_x, y)
    #axs[0, 1].set_title('Epoch vs. Training Accuracy')
    #axs[1, 0].plot(epoch_x, y)
    #axs[1, 0].set_title('Epoch vs. Test Loss')
    axs[0, 1].plot(epoch_x, test_accuracies)
    axs[0, 1].set_title('Epoch vs. Test Accuracy')
    plt.savefig(fig)
    
def evaluate(args, model, testloader):
    total_count = torch.tensor([0.0]).cuda(); correct_count = torch.tensor([0.0]).cuda()
    for i, data in enumerate(testloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

        total_count += labels.size(0)

        with torch.no_grad():
            cls_scores = model(imgs, with_dyn=args.with_dyn)

            predict = torch.argmax(cls_scores, dim=1)
            correct_count += (predict == labels).sum()
    testing_accuracy = correct_count / total_count
    return testing_accuracy.item()


def resume(args, model, optimizer):
    checkpoint_path = './{}_checkpoint.pth'.format(args.exp_id)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    ### load the model and the optimizer --------------------------------
    #
    #
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    #
    #
    ### -----------------------------------------------------------------

    print('Resume completed for the model\n')

    return model, optimizer
