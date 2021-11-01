import torch
import torch.nn as nn


class XSpkEmoTransLoss(nn.Module):
    """ XSpkEmoTrans Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(XSpkEmoTransLoss, self).__init__()
        self.alpha = train_config["loss"]["alpha"]
        self.beta = train_config["loss"]["beta"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            _,
            _,
            duration_targets,
            *_,
        ) = inputs[6:]
        (
            mel_iters,
            score_hard,
            score_soft,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            mel_lens,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        mel_targets.requires_grad = False
        score_hard.requires_grad = False
        mel_lens.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(
            src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # Iterative Loss
        mel_iter_loss = torch.zeros_like(mel_lens, dtype=mel_targets.dtype)
        for mel_iter in mel_iters:
            mel_iter_loss += self.mae_loss(mel_iter.masked_select(
                mel_masks.unsqueeze(-1)), mel_targets)
        mel_loss = (mel_iter_loss / (len(mel_iters) * mel_lens)).mean()

        emotion_classifier_loss = self.bce_loss(score_soft, score_hard)

        duration_loss = self.mse_loss(
            log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + self.alpha * emotion_classifier_loss + self.beta * duration_loss
        )

        return (
            total_loss,
            mel_loss,
            emotion_classifier_loss,
            duration_loss,
        )
