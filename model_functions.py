# define function to perform validation steps
def validation(model, criterion, dataloader):
    """
    This function takes in a model and validation dataloader as inputs 
    and calculates the loss and accuracy on the validation set.

    Inputs: model, dataloader
    Returns: valid_loss, accuracy


    """
    valid_loss = 0
    accuracy = 0
    for images, labels in dataloader:
        # move inputs to cuda
        images, labels = images.to('cuda'), labels.to('cuda')
        
        output = model.forward(images)
        
        loss = criterion(output, labels)
        valid_loss += loss.item()
        
        ps = torch.exp(output)
        predicted_label = ps.max(dim=1)[1]
        
        # compare output data with true labels (1 where equal, 0 where unequal)
        equality = (labels.data == predicted_label)
        # convert from byte tensor to float tensor to be able to take mean
        equality = equality.type(torch.FloatTensor)
        
        accuracy += equality.mean()
        
    return valid_loss, accuracy