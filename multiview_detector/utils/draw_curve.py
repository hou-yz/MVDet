# import matplotlib
#
# matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_curve(path, x_epoch, train_loss, train_prec, test_loss, test_prec, test_moda=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(131, title="loss")
    ax2 = fig.add_subplot(132, title="prec")
    ax1.plot(x_epoch, train_loss, 'bo-', label='train' + ': {:.3f}'.format(train_loss[-1]))
    ax1.plot(x_epoch, test_loss, 'ro-', label='test' + ': {:.3f}'.format(test_loss[-1]))
    ax2.plot(x_epoch, train_prec, 'bo-', label='train' + ': {:.1f}'.format(train_prec[-1]))
    ax2.plot(x_epoch, test_prec, 'ro-', label='test' + ': {:.1f}'.format(test_prec[-1]))

    ax1.legend()
    ax2.legend()
    if test_moda is not None:
        ax3 = fig.add_subplot(133, title="moda")
        ax3.plot(x_epoch, test_moda, 'ro-', label='test' + ': {:.1f}'.format(test_moda[-1]))
        ax3.legend()
    fig.savefig(path)
    plt.close(fig)
