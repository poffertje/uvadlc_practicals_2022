################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

"""Defines various kinds of visual-prompting modules for images."""
import torch
import torch.nn as nn
import numpy as np


class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Define the padding as variables self.pad_left, self.pad_right, self.pad_up, self.pad_down

        # Hints:
        # - Each of these are parameters that we need to learn. So how would you define them in torch?
        # - See Fig 2(c) in the assignment to get a sense of how each of these should look like.
        # - Shape of self.pad_up and self.pad_down should be (1, 3, pad_size, image_size)
        # - See Fig 2.(g)/(h) and think about the shape of self.pad_left and self.pad_right

        pad = torch.randn(1, 3, pad_size, image_size)
        self.pad_up = torch.nn.Parameter(pad)
        self.pad_down = torch.nn.Parameter(pad)

        height = image_size - 2*pad_size
        pad = torch.randn(1, 3, height, pad_size)
        self.pad_right = torch.nn.Parameter(pad)
        self.pad_left = torch.nn.Parameter(pad)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, add the prompt as a padding to the image.

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        # create a mask of the input batch and fill it with zeros
        mask = torch.zeros(x.shape)
        # add pad up
        padup_height = self.pad_up.shape[2]
        mask[:, :, 0:padup_height, :] = self.pad_up
        # add pad down
        image_height = x.shape[2]
        h_diff = image_height - padup_height
        mask[:, :, h_diff:image_height, :] = self.pad_down
        # add pad left
        padleft_width = self.pad_left.shape[3]
        mask[:, :, padup_height:h_diff, 0:padleft_width] = self.pad_left
        # add pad right
        image_width = x.shape[3]
        w_diff = image_width - padleft_width
        mask[:, :, padup_height:h_diff, w_diff:image_width] = self.pad_right

        prompt = mask + x

        return prompt
        #######################
        # END OF YOUR CODE    #
        #######################


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can define as self.patch) of size [prompt_size, prompt_size]
        # that is placed at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

        patch = torch.randn(1, 3, args.prompt_size, args.prompt_size)
        self.patch = torch.nn.Parameter(patch)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        # create a mask of the input batch and fill it with zeros
        mask = torch.zeros(x.shape)
        # get patch dimensions
        h, w = self.patch.shape[2], self.patch.shape[3]
        # put the patch over the mask in the top left corner
        mask[:, :, 0:h, 0:w] = self.patch
        # create the prompt by filling the zeros in the mask with the input batch pixel values
        prompt = mask + x

        return prompt

        #######################
        # END OF YOUR CODE    #
        #######################


class RandomPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a random patch in the image.
    For refernece, this prompt should look like Fig 2(b) in the PDF.
    """
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can be defined as self.patch) of size [prompt_size, prompt_size]
        # that is located at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

        patch = torch.randn(1, 3, args.prompt_size, args.prompt_size)
        self.patch = torch.nn.Parameter(patch)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - Note that, here, you need to place the patch at a random location
        #   and not at the top-left corner.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        # create a mask of the input batch and fill it with zeros
        mask = torch.zeros(x.shape)

        patch_h, patch_w = self.patch.shape[2], self.patch.shape[3]
        x_h, x_w = x.shape[2], x.shape[3]
        # create random path origin
        random_x = np.random.randint(0, x_w - patch_w)
        random_y = np.random.randint(0, x_h - patch_h)

        mask[:, :, random_x:random_x+patch_h, random_y:random_y+patch_w] = self.patch
        prompt = mask + x

        return prompt
        
        #######################
        # END OF YOUR CODE    #
        #######################

