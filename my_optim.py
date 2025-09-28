import math
import torch
import torch.optim as optim

class SharedRMSprop(optim.RMSprop):
    # Shared RMSProp (non-centered) for asynchronous RL (A3C).
    # - ���v g ���X���b�h�Ԃŋ��L���A�񓯊��ɍX�V
    # - �_���̕�� �uRMSProp�v�� (Eq. S2, S3) �ɑΉ�
    def __init__(self,
                 params,
                 lr=7e-4,           # �����̃� �w�K���̋L��
                 alpha=0.99          # decay factor �_��8�͂̐ݒ肻�̂܂�
                 ):
        super().__init__(params, lr=lr, alpha=alpha)

        # ���L�����ԃe���\�����m�ہi�S�p�����[�^�œ��T�C�Y�j
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # g = square_avg
                state.setdefault('g', torch.zeros_like(p.data))

    def share_memory(self):
        # ��ԓ��v�����L���������i�e�v���Z�X���瓯��o�b�t�@���Q�Ɓj
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # state['step'].share_memory_()
                state['g'].share_memory_()  # ��S2(g �� �� g + (1-��) (����)^2)��g�A���z�̓�敽��

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr      = group['lr']         # �_�������� ��
            alpha   = group['alpha']       # �_�������� ��
            eps     = 1e-8

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                g = state['g']  # ���ꂪ���L���Ă���x�N�g�� g

                # ��S2(g �� �� g + (1-��) (����)^2)�̎��s
                # WRITE ME
                g.mul_(alpha).addcmul_(grad, grad, value=(1.0 - alpha))

                # ��S3(�� �� �� - �� * ���� / sqrt(g+eps))�̎��s
                # WRITE ME
                s = g.sqrt().add_(eps)
                p.data.addcdiv_(grad, s, value=-lr)

        return loss
