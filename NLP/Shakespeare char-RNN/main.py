import tensorflow as tf

shakespeare_url = "https://homl.info/shakespeare"
def main():
    filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
    with open(filepath) as f:
        shakespeare_text = f.read()
    print(shakespeare_text[:80])

    text_vec_layer = tf.keras.layers.TextVectorizing(split="character", standardize="lower")
    text_vec_layer.adapt([shakespeare_text])
    encoded = text_vec_layer([shakespeare_text])[0]
    encoded -= 2 #tokens 0 und 1 auslassen
    n_tokens = text_vec_layer.vocabulary.size() -2 #Anzahl verschiedener Buchstaben = 39
    dataset_size = len(encoded)

def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    '''
    This function takes a sequence as input and creates a dataset containing all the windows of the desired length.
    It increases the length by one, since we need the next character for the target.
    Then, it shuffles the windows (optionally), batches them, splits them into input/output paris and activates prefetching
    :param sequence:
    :param length: Größe des Fensters
    :param shuffle:
    :param seed:
    :param batch_size:
    :return:
    '''
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length +1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length +1))
    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:,:-1],window[:,1:])).prefetch(1)


if __name__ == "__main__":
    main()