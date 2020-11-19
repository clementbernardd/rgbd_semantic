from importations import *
import pickle5 as pickle

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def convert_img_to_labels(img, to_plot = True ) :
    ''' Plot the img '''
    uniques = np.unique(img)
    dict_unique = {}
    new_img = img.copy()
    for i,unique in enumerate(uniques) :
        dict_unique[unique] =  i
    for x in dict_unique :
        new_img[new_img == x] = dict_unique[x]
    if to_plot :
        plt.imshow(new_img)
        plt.show()
    return new_img


def plot_multiple_images(images_dict, rows = 1 , cols = 1 ):
    ''' Plot the images from the dictionnary : {title : image} '''
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    ax = np.array(ax)
    for i,title in enumerate(images_dict) :
        ax.flatten()[i].imshow(images_dict[title])
        ax.flatten()[i].set_title(title)
        ax.flatten()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def one_hot_encode(y_train, N = 22) :
    ''' One hot encode the labels '''
    if y_train.shape == 2 :
        y_train = y_train.reshape(1, y_train.shape[0],y_train.shape[1])
    y_train_hot = np.zeros((y_train.shape[0], N, y_train.shape[1], y_train.shape[2]))

    for i,img in enumerate(y_train) :
        for j,x in enumerate(img) :
            for k,y in enumerate(x) :

                hot_vector = np.zeros((N,))
                hot_vector[int(y)] = 1
                y_train_hot[i,:,j,k] = hot_vector


    return y_train_hot.reshape(y_train.shape[0], N ,y_train.shape[1], y_train.shape[2])


def get_batch(X_train, y_train, idxs,current_batch , BATCH_SIZE ) :
    ''' Return the batch of X_train and y_train '''
    n = idxs.shape[0]
    indexes = idxs[current_batch : min(n, current_batch+BATCH_SIZE)]

    batch_x0 = X_train[0][indexes, : ]
    batch_x1 = X_train[1][indexes, :]

    batch_x = [ batch_x0, batch_x1 ]
    batch_y = y_train[indexes, :]

    return batch_x, batch_y




def one_hot_encode_torch(y_train, N = 23) :
    ''' One hot encode the labels. y_train is a torch tensor '''
    # assert(len(y_train.shape) == 3)

    result = one_hot_encode(y_train.detach().numpy())
    return torch.from_numpy(result)


def split_train_valid_test(nyu) :
    ''' Take the nyu object and return a split into train, validation and test '''
    images = nyu['Images'].reshape(-1, 3, 480, 640)
    depths = nyu['Depths'].reshape(-1,1, 480, 640)
    labels = nyu['Labels'].reshape(-1, 480, 640)
    N = np.max(labels)

#     labels_hot = one_hot_encode(labels, N)

    idx = np.arange(images.shape[0])
    np.random.shuffle(idx)

    train_ids = idx[:700]
    val_ids = idx[700:795]
    test_ids = idx[795:]

    images_train = images[train_ids]
    images_val = images[val_ids]
    images_test = images[test_ids]

    depth_train = depths[train_ids]
    depth_val = depths[val_ids]
    depth_test = depths[test_ids]

    label_train = labels[train_ids]
    label_val = labels[val_ids]
    label_test = labels[test_ids]

    train, val, test = {},{},{}

    train['images'] = images_train
    train['depths'] = depth_train
    train['labels'] = label_train

    val['images'] = images_val
    val['depths'] = depth_val
    val['labels'] = label_val

    test['images'] = images_test
    test['depths'] = depth_test
    test['labels'] = label_test

    return train, val, test



def get_train_val_test(dataset, is_torch = True ) :
    ''' Return X and Y from the dataset (nyu) '''
    X = [dataset['images'], dataset['depths']]
    Y = dataset['labels']
    if is_torch :

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X[0] = torch.from_numpy(X[0]).type(torch.float).to(device)
        X[1] = torch.from_numpy(X[1]).type(torch.float).to(device)
        Y =  torch.from_numpy(Y.astype(float)).to(device)

    return X,Y


def convert_hot_to_squeeze(img, N = N ) :
    ''' Convert an image of shape [N, x,y ] into [x, y] by taking the argmax '''
    return np.argmax(img, axis = 0)

def feature_map_size(in_ch  , K : int , k_shape : list , s : int , p : int ) :
    ''' Return the shape of the feature map size for a convolution filter
        Inputs :
            - img_shape : Shape of the input
            - K : number of kernels
            - k_shape : tuple for the size of the kernel
            - s : Stride
            - p : padding
    '''

    out_ch = in_ch.copy()

    out_ch[:-1] = (np.array(in_ch[:-1]) - np.array(k_shape) + 2 * p)/s + 1

    out_ch[-1] = K

    return np.array(out_ch).astype(int)



''' Pre-processing functions '''

def get_distribution(scenes) :
    ''' Return a dictionnary with the distribution of the category  '''
    categories = []

    for cat in scenes :
        categories.append(cat[0][0][0])

    distribution = {}

    for cat in np.unique(scenes) :

        count = np.where(np.array(categories) == cat[0][0]  )[0].shape[0]
        distribution[cat[0][0]] = count
    return distribution


def plot_sns(x,y, xlabel, ylabel, title) :
    ''' Plot into bar the inputs '''
    sns.set(rc={'figure.figsize':(12,8)})
    # Create a barplot for the distribution of the inputs
    ax = sns.barplot(x=x, y=y, ci=None)
    # Move the xlabels to be the name of the inputs
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    ax.grid(True)
    plt.title(title)
    plt.show()


def plot_histogram(scenes, params= None , is_plot = True  ) :
    ''' Plot the histogram of the distribution of the input '''
    distribution_scenes = get_distribution(scenes)
    distribution_scenes = dict(sorted(distribution_scenes.items(), key=lambda item: item[1]))
    scenes = list(distribution_scenes.keys())
    distribution = list(distribution_scenes.values())
    scenes.reverse()
    distribution.reverse()

    if is_plot :
        if params is None :
            plot_sns(scenes, distribution,'Scenes','Number of images in the dataset','Scenes distribution')
        else :
            plot_sns(scenes, distribution,**params )

    return distribution_scenes


def get_num_classes(images, all_scenes) :
    ''' Return the number of classes for the different scenes '''
    num_classes = {}
    scenes = []
    for scene in all_scenes :
        scenes.append(scene[0][0][0])
    # For every scene
    for scene in np.unique(scenes) :
        # Get the index where the scene is the current one
        idxs = np.where(np.array(scenes) == scene)[0]
        # If the dictionnary is empty for this scene
        if scene not in num_classes :
            # Create a set
            num_classes[scene] = set()
        # For all the labels of the given scene
        for idx in idxs :
            # Get the labels
            img = images[:,:,idx]
            # For all the labels in the image
            for classes in np.unique(img) :
                # Add the label to the set
                num_classes[scene].add(classes)
    # For every scene
    for key in num_classes :
        # Get the number of different labels
        num_classes[key] = len(num_classes[key])

    return num_classes


def plot_num_class(images, scenes, is_plot = True ) :
    '''  Plot the number of classes '''

    num_classes = get_num_classes(images, scenes)
    if is_plot :
        plot_sns(list(num_classes.keys()),\
                 list(num_classes.values()),'Scenes','Number of different classes','Labels distribution')

    return num_classes


def plot_classes_distribution(distribution_scenes,num_classes ) :
    ''' Plot the number of classes and distribution for the different scenes '''
    all_distribution = []

    for scene in distribution_scenes :

        dist = distribution_scenes[scene]

        n_class = num_classes[scene]
        all_distribution.append({'Scenes': scene,
                       'score': dist,
                       'marker': 'Number of images'})

        all_distribution.append({'Scenes': scene,
                       'score': n_class,
                       'marker': 'Number of classes'})


    df = pd.DataFrame(all_distribution)

    sns.set(rc={'figure.figsize':(14,20)})
    g = sns.catplot(x="Scenes", y="score", hue="marker", data=df,
                    kind="bar", height=12, aspect=2)


    g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")

    plt.axhline(20, ls='--', color = 'r', label = '20 classes')

    plt.legend()

    g.ax.grid(True)


def get_number_class_scene(images, all_scenes, current_scenes ) :
    ''' Return the number of classes for the mixed scenes, and the number of images   '''
    n_classes = set()
    n_images = set()

    scenes = []
    for scene in all_scenes :
        scenes.append(scene[0][0][0])
    for scene in current_scenes :
        # Get the index where the scene is the current one
        idxs = np.where(np.array(scenes) == scene)[0]

        for idx in idxs :
            # Get the labels
            img = images[:,:,idx]
            n_images.add(idx)
            # For all the labels in the image
            for classes in np.unique(img) :
                # Add the label to the set
                n_classes.add(classes)


    return {'n_classes' : len(n_classes), 'n_images' : len(n_images)}

def get_distribution_classes(labels) :
    ''' Return a plot with the distribution of the classes over the images '''
    classes = {}

    for i in range(labels.shape[-1]) :

        for classe in np.unique(labels[:,:,i]) :

            if classe in classes  :

                classes[classe].append(i)
            else :

                classes[classe] = [i]

    n_classes = {}

    for classe in classes :

        n_classes[classe] = len(classes[classe])

    # Sort the classes
    n_classes = dict(sorted(n_classes.items(), key=lambda item: -item[1]))

    return n_classes


def get_nyu_classes(nyu, classes, step = 2 ) :
    ''' Return the images,depths, labels and scenes for the images with the given classes
        - step : The subsample of image that we do
    '''
    subsample = {'images' : [], 'depths' : [], 'labels' : [], 'scenes' : []}

    N = nyu['Images'].shape[-1]

    for i in range(N) :

        label = nyu['Labels'][:,:, i]

        for classe in classes :

            if classe in np.unique(label) :
            # Add it to the dataset

                image = nyu['Images'][:,:,:,i]
                depth = nyu['Depths'][:,:,i].reshape(480, 640,1)
                scene = nyu['Scenes'][i][0][0][0]

                image = image[np.arange(0, image.shape[0],step) , :]
                image = image[:, np.arange(0, image.shape[1],step) , :]

                depth = depth[np.arange(0, depth.shape[0],step) , :]
                depth = depth[:, np.arange(0, depth.shape[1],step) ]

                label = label[np.arange(0, label.shape[0],step) , :]
                label = label[:, np.arange(0, label.shape[1],step) ]

                subsample['images'].append(image)
                subsample['depths'].append(depth)
                subsample['labels'].append(label)
                subsample['scenes'].append(scene)

                break

    subsample['images'] = np.array(subsample['images'] )
    subsample['depths'] = np.array(subsample['depths'] )
    subsample['labels'] = np.array(subsample['labels'] )




    return subsample


def convert_labels(labels, classes) :
    ''' Converth the labels to be only those of the classes '''
    # Squeeze the classes into a growing number
    new_classes={}
    for i,classe in enumerate(sorted(classes)) :
        new_classes[classe] = i

    N = len(new_classes)

    new_labels = []


    for label in labels :

        new_label = label.copy()

        for classe in np.unique(label) :

            if classe in new_classes :

                new_label[np.where(new_label == classe)] = new_classes[classe]
            else :
                new_label[np.where(new_label == classe)] = N+1


        new_labels.append(new_label)

    return np.array(new_labels)



def compare_images(nyu, nyu_, N = 777) :
    ''' Compare the two datasets '''

    label_nyu = convert_img_to_labels(nyu['Labels'][:,:,N], to_plot = False)
    label_nyu_ = convert_img_to_labels(nyu_['labels'][N,:], to_plot = False)

    images_nyu = nyu['Images'][:,:,:,N]
    images_nyu_ = nyu_['images'][N,:]

    depth_nyu = nyu['Depths'][:,:,N]
    depth_nyu_ = nyu_['depths'][N,:]

    comparison = {'Original Image RGB (shape : {})'.format(images_nyu.shape) : images_nyu,\
                  'Subsampled Image RGB (shape : {})'.format(images_nyu_.shape) : images_nyu_, \
                 'Original Image Depth (shape : {})'.format(depth_nyu.shape) : depth_nyu, \
                  'Subsampled Image Depth (shape : {})'.format(depth_nyu_.shape) : depth_nyu_,\
                 'Original labels (shape : {})'.format(label_nyu.shape) : label_nyu,\
                  'Subsampled labels (shape : {})'.format(label_nyu_.shape): label_nyu_}

    plot_multiple_images(comparison, rows=3, cols=2)
