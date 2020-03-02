from torch import nn


class FTModel(nn.Module):
    def __init__(
            self,
            pretrained,
            layers_to_remove,
            num_features,
            num_classes,
            input_size=3,
            train_only_fc=False
    ):
        super(FTModel, self).__init__()

        # extract number of features in the last layer before the ones we remove
        in_features = list(pretrained.children())[-layers_to_remove].in_features

        # build the new model
        old_layers = list(pretrained.children())[:-layers_to_remove]
        self.new_model = nn.Sequential(*old_layers)
        self.fc = nn.Linear(in_features, num_features)
        self.drop = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(num_features, num_classes)

        self.train_only_fc = train_only_fc
        if self.train_only_fc:
            for params in self.new_model.parameters():
                params.requires_grad = False

    def forward(self, x):
        x = self.new_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        embs = x
        x = self.fc2(x)
        return x, embs


class FTModelEf(nn.Module):
    def __init__(
            self,
            pretrained,
            num_classes,
    ):
        super(FTModelEf, self).__init__()

        # build the new model
        self.features = pretrained
        self.fc = nn.Linear(pretrained._fc.out_features, num_classes)

    def forward(self, x):
        embs = self.features(x)
        x = self.fc(embs)
        return x, embs
