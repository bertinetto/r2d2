from PIL import Image


def to_pil(crop):
    # crop = crop.cpu().mul(255).clamp(0, 255).byte()
    crop = crop.cpu()[0, :, :].mul(255).byte()
    crop = crop.numpy()
    return Image.fromarray(crop)


def visualize_batch(x, n_way, n_shot, n_query):
    xs = x[:n_way * n_shot]
    xq = x[n_way * n_shot:]
    assert xq.size(0) == n_way * n_query

    for i in range(xs.size(0)):
            filename = 'out/example_%02d.bmp' % i
            support = to_pil(xs[i, :, :].data)
            support.save(filename)

    for i in range(xq.size(0)):
            filename = 'out/query_%02d.bmp' % i
            support = to_pil(xq[i, :, :].data)
            support.save(filename)
