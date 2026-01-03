import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
import os
import datasetClass
from PIL import Image
import copy
from Fed import FedAvg
import random
import function
import RS64
import generate_rscode
import time









def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Prepare datasets and transforms
    # -----------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # -----------------------------
    # Define model
    # -----------------------------
    net = ResNet18().to(device)

    # -----------------------------
    # Load CIFAR10 test dataset
    # -----------------------------
    testset = torchvision.datasets.CIFAR10(root=args.path_data_root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    # -----------------------------
    # Load client train dataset
    # -----------------------------
    total_train_dataset = torch.load(args.path_data)
    total_client_train_data = total_train_dataset[0]
    total_client_train_label = total_train_dataset[1]
    client_Labels = total_train_dataset[2]
    client_data_num = total_train_dataset[4]

    client_num = args.client_num

    client_train_dataset = [
        torch.utils.data.TensorDataset(torch.tensor(total_client_train_data[cn]),
                                       torch.LongTensor(total_client_train_label[cn]))
        for cn in range(client_num)
    ]
    client_trainloader = [
        torch.utils.data.DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True)
        for dataset in client_train_dataset
    ]

    # -----------------------------
    # Generate RS codes for clients
    # -----------------------------
    RS_n = args.RS_n
    RS_k = args.RS_k
    client_code = generate_rscode.code_lib(RS_k, RS_n, client_num)
    i_to_c = args.i_to_c
    image_num = i_to_c * RS_n

    client_class = []
    for inum in range(client_num):
        temp_class = []
        for ic in range(RS_n):
            temp = [int(digit) for digit in f"{client_code[inum][ic]:02d}"]
            for te in range(i_to_c):
                temp_class.append(temp[te])
        client_class.append(temp_class)

    # -----------------------------
    # Prepare trigger datasets
    # -----------------------------
    client_trigger_data = []
    for cn in range(client_num):
        images = []
        labels = []
        for image_ID in range(1, image_num + 1):
            img = Image.open(args.path_image.format(image_ID))
            images.append(img)
            labels.append(client_class[cn][image_ID - 1])
        client_trigger_data.append({'data': images, 'labels': labels})

    client_trigger_dataset = [
        datasetClass.customDataset(data_dict=client_trigger_data[cn], transform=transform_test)
        for cn in range(client_num)
    ]

    # -----------------------------
    # Utility functions
    # -----------------------------
    def testAcc(model):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        return correct / total

    def is_rs_restore(labels, rs_code, rs_n, rs_k):
        nk = rs_n - rs_k
        temp_code = []
        for i in range(0, len(labels), i_to_c):
            temp = [labels[i + j] for j in range(i_to_c)]
            temp_code.append(int("".join(map(str, temp))))
        restore_code = RS64.rs_correct_msg(temp_code, nk)
        return restore_code == rs_code

    def watermarkAcc(net, client_id):
        count = 0
        for id in range(image_num):
            image = Image.open(args.path_image.format(id + 1))
            input_image = transform_test(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = net(input_image)
            _, predicted = torch.max(output.data, 1)
            if predicted.item() == client_class[client_id][id]:
                count += 1
        return count / image_num

    def tracking_Acc(client_class, client_num, client_model, WSR_Threshold):
        leak_id = random.randint(0, client_num - 1)
        leak_class = client_class[leak_id]
        WSR = []
        c_labels = []
        for cn in range(client_num):
            count = 0
            temp_label = []
            for image_id in range(image_num):
                img = Image.open(args.path_image.format(image_id + 1))
                input_image = transform_test(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = client_model[cn](input_image)
                _, predicted = torch.max(output.data, 1)
                result = predicted.item()
                temp_label.append(result)
                if result == leak_class[image_id]:
                    count += 1
            WSR.append(count / image_num)
            c_labels.append(temp_label)

        max_and_second = function.found_largest_and_second(WSR)
        max_WSR_index, max_WSR = max_and_second[0]
        sec_WSR_index, sec_WSR = max_and_second[1]
        WSR_gap = max_WSR - sec_WSR

        with open(args.path_track_log, "a") as ff:
            ff.write(f"\nLeaking client: {leak_id + 1}")
            ff.write(f"\nMax WSR: client {max_WSR_index + 1}, WSR={max_WSR * 100:.3f}%")
            ff.write(f"\nSecond Max WSR: client {sec_WSR_index + 1}, WSR={sec_WSR * 100:.3f}%")
            ff.write(f"\nGap: {WSR_gap:.3f}%")
            if WSR_gap >= WSR_Threshold and max_WSR_index == leak_id:
                if is_rs_restore(c_labels[max_WSR_index], client_code[max_WSR_index], RS_n, RS_k):
                    ff.write("\nTracking succeeded and RS code restored.\n")
                else:
                    ff.write("\nTracking succeeded but RS code not restored.\n")
            else:
                ff.write("\nTracking failed.\n")

    def combined_loss(output_task2, target_task2, model_task2, model_task1, beta):
        loss_task2 = nn.CrossEntropyLoss()(output_task2, target_task2)
        l2_norm = sum(torch.norm(p2 - p1, p=2) ** 2 for p1, p2 in zip(model_task1.parameters(), model_task2.parameters()))
        return loss_task2 + beta * l2_norm

    # -----------------------------
    # Initialize global and local models
    # -----------------------------
    global_model = copy.deepcopy(net)
    global_weight = global_model.state_dict()
    local_model = [copy.deepcopy(global_model) for _ in range(client_num)]
    local_weight = [global_weight for _ in range(client_num)]
    client_beta = [args.ori_beta for _ in range(client_num)]

    total_ori_time = 0
    total_wat_time = 0

    # -----------------------------
    # Start federated training
    # -----------------------------
    for iter in range(args.iterate_epoch):
        print("Origin task training")
        ori_time_s = time.time()
        for cn in range(client_num):
            local_model[cn].load_state_dict(local_weight[cn])
            optimizer = optim.SGD(local_model[cn].parameters(), lr=args.LR, momentum=0.9, weight_decay=5e-4)

            for epoch in range(args.EPOCH):
                print(f"Iter {iter+1}, Client {cn+1}, Epoch {epoch+1}")
                local_model[cn].train()
                sum_loss = 0.0
                correct = 0
                total = 0
                for i, (inputs, labels) in enumerate(client_trainloader[cn]):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model[cn](inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print(f"[Client {cn+1}, Epoch {epoch+1}, Batch {i+1}] Loss: {sum_loss/(i+1):.3f} | Acc: {100.*correct/total:.3f}%")

            # Save client accuracy
            with open(args.path_origin_client_log, "a") as f1:
                f1.write(f"\nClient {cn+1}, Iter {iter+1}, Test Acc: {100.*testAcc(local_model[cn]):.3f}%")
                f1.write(f"\nClient {cn+1}, Iter {iter+1}, Watermark Acc: {100.*watermarkAcc(local_model[cn], cn):.3f}%")

        ori_time_e = time.time()
        ori_time = ori_time_e - ori_time_s
        total_ori_time += ori_time
        with open(args.path_time_log, "a") as f1:
            f1.write(f'Iter {iter+1} origin task duration: {ori_time}\n')

        # -----------------------------
        # Aggregate global model
        # -----------------------------
        local_weight = [m.state_dict() for m in local_model]
        global_weight = FedAvg(local_weight)
        global_model.load_state_dict(global_weight)
        for cn in range(client_num):
            local_model[cn].load_state_dict(global_weight)

        with open(args.path_origin_global_log, "a") as f2:
            f2.write(f"\nIter {iter+1}, Aggregated Global Test Acc: {100.*testAcc(global_model):.3f}%")

        # -----------------------------
        # Watermark insertion
        # -----------------------------
        print("Watermarking")
        wat_time_s = time.time()
        for cn in range(client_num):
            w_trainloader = torch.utils.data.DataLoader(client_trigger_dataset[cn],
                                                        batch_size=args.w_BATCH_SIZE, shuffle=False)
            epoch = 0
            w_acc = 0
            temp_model = copy.deepcopy(local_model[cn])
            optimizer = optim.SGD(temp_model.parameters(), lr=args.w_LR, momentum=0.9, weight_decay=5e-4)
            beta = client_beta[cn] + args.beta_ad

            while w_acc < 1:
                temp_model.train()
                for i, (inputs, labels) in enumerate(w_trainloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = temp_model(inputs)
                    loss = combined_loss(outputs, labels, temp_model, global_model, beta)
                    loss.backward()
                    optimizer.step()
                w_acc = watermarkAcc(temp_model, cn)
                epoch += 1
                if epoch > args.max_epoch:
                    temp_model = copy.deepcopy(local_model[cn])
                    optimizer = optim.SGD(temp_model.parameters(), lr=args.w_LR, momentum=0.9, weight_decay=5e-4)
                    w_acc = 0
                    epoch = 0
                    beta -= args.beta_re

            local_model[cn] = copy.deepcopy(temp_model)
            client_beta[cn] = beta
            local_weight[cn] = local_model[cn].state_dict()
            with open(args.path_watermark_client_log, "a") as f3:
                f3.write(f"\nClient {cn+1}, Iter {iter+1}, Watermark inserted epochs: {epoch}")
                f3.write(f"\nClient {cn+1}, Iter {iter+1}, Final beta: {beta}")
                f3.write(f"\nClient {cn+1}, Iter {iter+1}, Post-watermark Test Acc: {100.*testAcc(local_model[cn]):.3f}%")
                f3.write(f"\nClient {cn+1}, Iter {iter+1}, Post-watermark Watermark Acc: {100.*watermarkAcc(local_model[cn], cn):.3f}%")

        wat_time_e = time.time()
        total_wat_time += wat_time_e - wat_time_s
        with open(args.path_time_log, "a") as f1:
            f1.write(f'Iter {iter+1} watermark insertion duration: {wat_time_e - wat_time_s}\n')

        # -----------------------------
        # Tracking evaluation
        # -----------------------------
        with open(args.path_track_log, "a") as ff:
            ff.write(f"\nTracking result Iter {iter+1}:")
        tracking_Acc(client_class, client_num, local_model, args.WSR_Threshold)

    # -----------------------------
    # Save final models
    # -----------------------------
    torch.save(global_model.state_dict(), args.path_global_model)
    for client in range(len(local_model)):
        torch.save(local_model[client].state_dict(), args.path_client_model.format(client + 1))

    with open(args.path_time_log, "a") as f1:
        f1.write(f'\nTotal origin task time: {total_ori_time}\nTotal watermark insertion time: {total_wat_time}\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with Watermarking")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--path_data_root", type=str, required=True)
    parser.add_argument("--path_data", type=str, required=True)
    parser.add_argument("--path_image", type=str, required=True)
    parser.add_argument("--path_track_log", type=str, required=True)
    parser.add_argument("--path_it_log", type=str, required=True)
    parser.add_argument("--path_origin_client_log", type=str, required=True)
    parser.add_argument("--path_origin_global_log", type=str, required=True)
    parser.add_argument("--path_watermark_client_log", type=str, required=True)
    parser.add_argument("--path_global_model", type=str, required=True)
    parser.add_argument("--path_client_model", type=str, required=True)

    parser.add_argument("--ori_beta", type=float, default=0.1)
    parser.add_argument("--beta_ad", type=float, default=0.02)
    parser.add_argument("--beta_re", type=float, default=0.02)
    parser.add_argument("--max_epoch", type=int, default=15)

    parser.add_argument("--EPOCH", type=int, default=5)
    parser.add_argument("--BATCH_SIZE", type=int, default=128)
    parser.add_argument("--LR", type=float, default=0.01)

    parser.add_argument("--w_BATCH_SIZE", type=int, default=1)
    parser.add_argument("--w_LR", type=float, default=0.001)

    parser.add_argument("--iterate_epoch", type=int, default=60)
    parser.add_argument("--WSR_Threshold", type=float, default=0.5)

    parser.add_argument("--client_num", type=int, default=20)
    parser.add_argument("--RS_n", type=int, default=62)
    parser.add_argument("--RS_k", type=int, default=2)
    parser.add_argument("--i_to_c", type=int, default=2)

    args = parser.parse_args()
    main(args)