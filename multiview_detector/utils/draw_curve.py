# import matplotlib
#
# matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_curve(path, x_epoch, train_loss, train_prec, og_test_loss, og_test_prec,
               masked_test_loss=None, masked_test_prec=None, prec_labels=['train', 'test (og)', 'test (masked)'],
               loss_labels=None):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="prec")
    if loss_labels is None:
        loss_labels = prec_labels
    ax0.plot(x_epoch, train_loss, 'bo-', label=loss_labels[0] + ': {:.3f}'.format(train_loss[-1]))
    ax1.plot(x_epoch, train_prec, 'bo-', label=prec_labels[0] + ': {:.3f}'.format(train_prec[-1]))
    ax0.plot(x_epoch, og_test_loss, 'ro-', label=loss_labels[1] + ': {:.3f}'.format(og_test_loss[-1]))
    ax1.plot(x_epoch, og_test_prec, 'ro-', label=prec_labels[1] + ': {:.3f}'.format(og_test_prec[-1]))
    if masked_test_loss is not None:
        ax0.plot(x_epoch, masked_test_loss, 'go-', label=loss_labels[2] + ': {:.3f}'.format(masked_test_loss[-1]))
    if masked_test_prec is not None:
        ax1.plot(x_epoch, masked_test_prec, 'go-', label=prec_labels[2] + ': {:.3f}'.format(masked_test_prec[-1]))

    ax0.legend()
    ax1.legend()
    fig.savefig(path)
    plt.close(fig)
