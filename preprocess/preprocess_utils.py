from flow_data_preprocess import build_flow_data
from packet_data_preprocess import build_packet_data
import random
import json
import os
from tqdm import tqdm
import math
from transformers import AutoTokenizer

TRAINING_SAMPLE_RATIO = 0.9


def split_dataset(build_data, max_sampling_number, sampling=True):
    print(len(build_data))
    random.shuffle(build_data)
    if sampling is True:
        train_nb = int(min(max_sampling_number, len(build_data)) * TRAINING_SAMPLE_RATIO)
        test_nb = int(min(max_sampling_number, len(build_data)) * (1 - TRAINING_SAMPLE_RATIO))
    else:
        train_nb = int(len(build_data) * TRAINING_SAMPLE_RATIO)
        test_nb = int(len(build_data) * (1 - TRAINING_SAMPLE_RATIO))

    train_data = build_data[:train_nb]
    test_data = build_data[train_nb:train_nb + test_nb]

    return train_data, test_data


def write_dataset(dataset, output_path):
    random.shuffle(dataset)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fin:
        for data in dataset:
            json.dump(data, fin)
            fin.write("\n")
        # json.dump(dataset, fin, indent=4, separators=(',', ': '))


def write_labels(labels, output_path):
    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i
    with open(output_path, "w", encoding="utf-8") as fin:
        json.dump(label_dict, fin, indent=4, separators=(',', ': '))


def build_data_from_dir(args, files_path, sampling_method=None, max_sampling_number=None):
    build_data = []
    pcaps = os.listdir(files_path)
    samples_per_pcap = -1
    if sampling_method == "average_sampling" and max_sampling_number is not None:
        samples_per_pcap = math.ceil(max_sampling_number / len(pcaps))
    max_packet_number = getattr(args, "max_packet_number", None)
    max_token_length = getattr(args, "max_token_length", None)
    tokenizer_path = getattr(args, "tokenizer_path", None)
    tokenizer = None
    enable_packet_loop = (
        args.granularity == "flow"
        and max_packet_number is not None
        and max_token_length is not None
    )
    if enable_packet_loop:
        if tokenizer_path is None:
            raise ValueError("tokenizer_path is required when max_token_length is set")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def exceeds_token_limit(flow_data):
        if not flow_data:
            return False
        prompt_prefix = getattr(args, "prompt_prefix", "") or ""
        prompt_suffix = getattr(args, "prompt_suffix", "") or ""
        max_len = max(
            len(
                tokenizer(
                    f"{prompt_prefix}{item}{prompt_suffix}",
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                ).input_ids
            )
            for item in flow_data
        )
        return max_len > max_token_length

    for pcap in tqdm(pcaps):
        if args.granularity == "flow":
            if args.max_sampling_number is not None and len(build_data) >= args.max_sampling_number:
                break
            if enable_packet_loop:
                current_max = max(1, max_packet_number)
                invalid_flag = False
                while True:
                    if current_max <= 0:
                        invalid_flag = True
                        break
                    pcap_data = build_flow_data(
                        os.path.join(files_path, pcap),
                        args.flow_feature,
                        max_packet_number=current_max,
                    )
                    if not exceeds_token_limit(pcap_data):
                        break
                    current_max -= 1
                if invalid_flag:
                    print("Invalid flow data: ", pcap)
                    continue
            elif max_packet_number is not None:
                pcap_data = build_flow_data(
                    os.path.join(files_path, pcap),
                    args.flow_feature,
                    max_packet_number=max_packet_number,
                )
            else:
                pcap_data = build_flow_data(os.path.join(files_path, pcap), args.flow_feature)
        else:
            pcap_data = build_packet_data(os.path.join(files_path, pcap), samples_per_pcap=samples_per_pcap)
        build_data.extend(pcap_data)
    return build_data


def build_dataset(args, path, file, sampling_method=None):
    files_path = os.path.join(path, file)
    build_data = build_data_from_dir(
        args,
        files_path,
        sampling_method=sampling_method,
        max_sampling_number=args.max_sampling_number,
    )

    train_data, test_data = split_dataset(build_data=build_data, max_sampling_number=args.max_sampling_number)
    return train_data, test_data


def build_dataset_from_split(args, path, file):
    files_path = os.path.join(path, file)
    return build_data_from_dir(args, files_path)


def save_dataset(args, train_dataset, test_dataset, val_dataset=None):
    write_dataset(train_dataset, os.path.join(args.output_path, args.output_name + "_" + args.traffic_task + "_" +
                                              args.granularity + "_train.json"))
    write_dataset(test_dataset, os.path.join(args.output_path, args.output_name + "_" + args.traffic_task + "_" +
                                             args.granularity + "_test.json"))
    if val_dataset is not None:
        write_dataset(val_dataset, os.path.join(args.output_path, args.output_name + "_" + args.traffic_task + "_" +
                                                 args.granularity + "_val.json"))


def get_td_prompt_components(task_name, granularity, second_label=None):
    if task_name == "MBD":  # Mobile Behavior Detection
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                      "traffic features, and payloads. Please conduct the MOBILE BEHAVIOR DETECTION TASK to determine " \
                      "which type of mobile behavior the traffic belongs to. The categories " \
                      "include 'sendText, sendAudio, sendImage, shareLocationOnce, transferFile'."
        output = second_label

    elif task_name == "EMD":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                     "traffic features, and payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine " \
                     "which application category the encrypted beign or malicious traffic belongs to. The categories " \
                     "include 'BitTorrent, FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, WorldOfWarcraft," \
                     "Cridex, Geodo, Htbot, Miuref, Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus'."

        output = second_label

        # instruction = "Below is a traffic " + granularity + ". Please conduct the encrypted malware detection task: "
        #
        # output = "This might be a " + first_label + \
        #          " traffic " + granularity + ". The category is likely to be recognized as " + second_label + "."

    elif task_name == "EAC":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                     "traffic features, and payloads. Please conduct the ENCRYPTED APP CLASSIFICATION TASK to determine " \
                     "which APP category the encrypted traffic belongs to. The categories " \
                     "include '163Mail, 51cto, Acm, Adobe, Alibaba, Alicdn, Alipay, Amap, AmazonAWS, AmpProject, Apple," \
                     "Arxiv, Asus, Atlassian, AzureEdge, Baidu, Bilibili, Biligame, Booking, LA'."

        output = second_label
        # instruction = "Below is a traffic " + granularity + ". Please conduct the encrypted App classification task: "
        #
        # output = "The traffic category is likely to be recognized as " + second_label + "."

    elif task_name == "EAC2":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                     "traffic features, and payloads. Please conduct the ENCRYPTED APP CLASSIFICATION TASK to determine " \
                     "which APP category the encrypted traffic belongs to. The categories " \
                     "include '163, 51la, 51cto, Acm, Adobe, Alibaba, Alicdn, Alipay, Amap, AmazonAWS, AmpProject, Apple, " \
                     "Arxiv, Asus, Atlassian, AzureEdge, Baidu, Bilibili, Biligame, Booking, Chia, Chinatax, Cisco, Cloudflare, " \
                     "Cloudfront, Cnblogs, Codepen, Crazyegg, Criteo, Ctrip, Dailymotion, Deepl, Digitaloceanspaces, Duckduckgo, " \
                     "Eastday, Eastmoney, Elsevier, Facebook, Feishu, Ggpht, Github, Gitlab, Gmail, Goat, Google, Grammarly, " \
                     "Gravatar, Guancha, Huanqiu, Huawei, Hubspot, Huya, Ibm, Icloud, Ieee, Instagram, Iqiyi, Jb51, Jd, Kugou, " \
                     "LeetcodeCn, Media, Mi, Microsoft, Mozilla, Msn, Naver, Netflix, Nike, Notion, Nvidia, Office, Onlinedown, " \
                     "Opera, Oracle, Outbrain, Overleaf, Paypal, Pinduoduo, Python, Qcloud, Qq, Researchgate, Runoob, Sciencedirect, " \
                     "Semanticscholar, Sina, Smzdm, Snapchat, Sohu, Spring, Springer, Squarespace, Statcounter, Steampowered, " \
                     "Tco, Taboola, Teads, Thepaper, Tiktok, Toutiao, Twimg, Twitter, Unity3d, V2ex, Vivo, Vk, Vmware, Walmart, " \
                     "Weibo, Wikimedia, Wikipedia, Wp, Xiaomi, Ximalaya, Yahoo, Yandex, Youtube, Yy, Zhihu'."

        output = second_label
        # instruction = "Below is a traffic " + granularity + ". Please conduct the encrypted App classification task: "
        #
        # output = "The traffic category is likely to be recognized as " + second_label + "."

    elif task_name == "BND":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                      "traffic features, and payloads. Please conduct the BOTNET DETECTION TASK to determine " \
                      "which type of network the traffic belongs to. The categories " \
                      "include 'IRC, Neris, RBot, Virut, normal'."

        output = second_label
        # instruction = "Below is a traffic " + granularity + ". Please conduct the botnet detection task: "
        #
        # output = "The traffic category is likely to be recognized as " + second_label + "."

    elif task_name == "EVD":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                     "traffic features, and payloads. Please conduct the TRAFFIC DETECTION TASK to determine " \
                     "which behavior or application category the encrypted traffic belongs to. The categories " \
                     "include 'aim, bittorrent, email, gmail, facebook, ftps, hangouts, icq, netflix, scp, skype, spotify, " \
                     "tor, torrent, vimeo, voipbuster, vpn-ftps, vpn-sftp, youtube'."

        output = second_label

        # instruction = "Below is a traffic " + granularity + ". Please conduct the encrypted VPN detection task: "
        #
        # output = "The traffic category is likely to be recognized as " + second_label + "."

    elif task_name == "EVD2":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                     "traffic features, and payloads. Please conduct the TRAFFIC DETECTION TASK to determine " \
                     "which behavior or application category the encrypted traffic belongs to. " \
                     "Some categories are traffic transmitted via VPN，while others are not. The categories " \
                     "include 'chat, emial, ft, p2p, stream, voip, vpn-chat, vpn-email, vpn-ft, vpn-p2p, " \
                     "vpn-stream, vpn-voip'."

        output = second_label

    elif task_name == "MDD":
        instruction = "Below is a traffic " + granularity + ". Please conduct the malicious DoH detection task: "
        output = ""
        if second_label:
            output = "The traffic category is likely to be recognized as " + second_label + "."

    elif task_name == "TBD":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                     "traffic features, and payloads. Please conduct the TOR BEHAVIOR DETECTION TASK to determine " \
                     "which behavior or application category the traffic belongs to under the Tor network. " \
                     "The categories include 'audio, browsing, chat, file, mail, p2p, video, voip'."

        output = second_label

    elif task_name == "APT":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                                                                          "traffic features, and payloads. Please conduct the APT DETECTION TASK to determine " \
                                                                          "which behavior or application category the traffic belongs to under the APT attacks. " \
                                                                          "The categories include 'APT and normal'."

        output = second_label

        # instruction = "Below is a traffic " + granularity + ". Please conduct the Tor behavior detection task: "
        #
        # output = "The traffic category is likely to be recognized as " + second_label + "."

    # elif task_name == "ATD":
    #     instruction = "Below is a traffic " + granularity + ". Please conduct the adware traffic detection task: "
    #
    #     output = "The traffic category is likely to be recognized as " + second_label + "."
    #
    # elif task_name == "RTD":
    #     instruction = "Below is a traffic " + granularity + ". Please conduct the ransomware traffic detection task: "
    #
    #     output = "The traffic category is likely to be recognized as " + second_label + "."
    #
    # elif task_name == "STD":
    #     instruction = "Below is a traffic " + granularity + ". Please conduct the scareware traffic detection task: "
    #
    #     output = "The traffic category is likely to be recognized as " + second_label + "."

    return instruction, output


def get_td_prompt_prefix_suffix(task_name, granularity):
    instruction, _ = get_td_prompt_components(
        task_name=task_name,
        granularity=granularity,
        second_label="",
    )
    return f"{instruction}\n<{granularity}>: ", ""


def build_td_text_dataset(traffic_data, first_label=None, second_label=None, task_name=None, granularity=None):
    """Building the text datasets of traffic detection task"""
    instruction, output = get_td_prompt_components(
        task_name=task_name,
        granularity=granularity,
        second_label=second_label,
    )

    dataset = []
    for data in traffic_data:
        dataset.append(
            {
                "instruction": instruction + "\n<" + granularity + ">: " + data,
                "output": output
            }
        )

    return dataset


def build_tg_text_dataset(traffic_data, traffic_label, granularity=None):
    """Building the text datasets of traffic generation task"""
    instruction = "Please generate a " + granularity + " of " + traffic_label + " traffic."

    dataset = []
    for data in traffic_data:
        dataset.append(
            {
                "instruction": instruction,
                "output": data
            }
        )

    return dataset


def build_tu_text_dataset(traffic_data, fields=None):
    """Building the text datasets of traffic understanding task"""

    knowledge_fields = []
    api_calls = []

    if "IP" in fields:
        knowledge_fields.extend(
            ["IP Version", "IP Header Length", "Differentiated Services Field", "Total Length",
             "Identification", "IP Flags", "Fragment Offset", "Time to Live", "Protocol", "IP Header Checksum",
             "Source Address", "Destination Address"]
        )
        api_calls.extend(
            ["scapy-IP-version", "scapy-IP-ihl", "scapy-IP-tos", "scapy-IP-len", "scapy-IP-id", "scapy-IP-flags",
             "scapy-IP-frag", "scapy-IP-ttl", "scapy-IP-proto", "scapy-IP-chksum", "scapy-IP-src", "scapy-IP-dst"]
        )

    if "TCP" in fields:
        knowledge_fields.extend(
            ["Source Port", "Destination Port", "Sequence Number", "Acknowledge Number",
             "TCP Flags", "Window", "TCP Header Checksum", "Urgent Pointer", "Destination Address"]
        )
        api_calls.extend(
            ["scapy-TCP-sport", "scapy-TCP-dport", "scapy-TCP-seq", "scapy-TCP-ack", "scapy-TCP-flags",
             "scapy-TCP-window", "scapy-TCP-chksum", "scapy-TCP-urgptr", "scapy-TCP-options"]
        )

    if "UDP" in fields:
        knowledge_fields.extend(
            ["Source Port", "Destination Port", "UDP Length", "UDP Header Checksum"]
        )
        api_calls.extend(
            ["scapy-UDP-sport", "scapy-UDP-dport", "scapy-UDP-len", "scapy-UDP-chksum"]
        )

    if "TLS" in fields:
        knowledge_fields.extend(
            ["Content Type", "Record Version", "TLS Message", "Message Type", "Handshake Version",
             "Cipher Suites", "Extensions"]
        )
        api_calls.extend(
            ["scapy-TLS-type", "scapy-TLS-version", "scapy-TLS-msg", "scapy-TLS-msg-msgtype", "scapy-TLS-msg-version",
             "scapy-TLS-msg-ciphers", "scapy-TLS-msg-ext"]
        )

    if "DNS" in fields:
        knowledge_fields.extend(
            ["Transaction ID", "Response", "Opcode", "Authoritative", "Truncated", "Recursion Desired",
             "Recursion Available", "Z", "Answer Authenticated", "Non-Authenticated", "Questions", "Answer RRs",
             "Authority RRs", "Additional RRs", "Queries", "Answers"]
        )
        api_calls.extend(
            ["scapy-DNS-id", "scapy-DNS-qr", "scapy-DNS-opcode", "scapy-DNS-aa", "scapy-DNS-tc", "scapy-DNS-rd",
             "scapy-DNS-ra", "scapy-DNS-z", "scapy-DNS-ad", "scapy-DNS-cd", "scapy-DNS-qdcount", "scapy-DNS-ancount",
             "scapy-DNS-nscount", "scapy-DNS-arcount", "scapy-DNS-qd", "scapy-DNS-an"]
        )

    if "http.HTTPRequest" in fields:
        knowledge_fields.extend(
            ["Headers", "Host", "User-Agent", "Accept", "Connection", "Method", "Path", "Http-Version", "Range",
             "Accept-Language", "Additional-Headers"]
        )
        api_calls.extend(
            ["scapy-http.HTTPRequest-Headers", "scapy-http.HTTPRequest-Host", "scapy-http.HTTPRequest-User-Agent",
             "scapy-http.HTTPRequest-Accept", "scapy-http.HTTPRequest-Connection", "scapy-http.HTTPRequest-Method",
             "scapy-http.HTTPRequest-Path", "scapy-http.HTTPRequest-Http-Version", "scapy-http.HTTPRequest-Range",
             "scapy-http.HTTPRequest-Accept-Language", "scapy-http.HTTPRequest-Additional-Headers"]
        )

    if "http.HTTPResponse" in fields:
        knowledge_fields.extend(
            ["Headers", 'Accept-Ranges', 'Server', 'Cache-Control', 'Connection', 'Date', 'Content-Length',
             'Content-Range', 'Content-Type', 'Last-Modified', 'Additional-Headers', 'Status-Line']
        )
        api_calls.extend(
            ["scapy-http.HTTPResponse-Headers", "scapy-http.HTTPResponse-Accept-Ranges",
             "scapy-http.HTTPResponse-Server", "scapy-http.HTTPResponse-Cache-Control",
             "scapy-http.HTTPResponse-Connection", "scapy-http.HTTPResponse-Date",
             "scapy-http.HTTPResponse-Content-Length", "scapy-http.HTTPResponse-Content-Range",
             "scapy-http.HTTPResponse-Content-Type", "scapy-http.HTTPResponse-Last-Modified",
             "scapy-http.HTTPResponse-Additional-Headers", "scapy-http.HTTPResponse-Status-Line"]
        )

    if "GeoIP" in fields:
        knowledge_fields.extend(
            ["source address", "destination address"]
        )
        api_calls.extend(
            ["<geoip-src>", "<geoip-dst>"]
        )

    if "JA3" in fields:
        knowledge_fields.extend(
            ["client fingerprints", "server fingerprints"]
        )
        api_calls.extend(
            ["<ja3-client>", "<ja3-server>"]
        )

    dataset = []

    for data in traffic_data:
        index = random.randint(0, len(knowledge_fields) - 1)
        if "GeoIP" in fields or "JA3" in fields:
            dataset.append(
                {
                    "instruction": "Please analyze the " + knowledge_fields[index] + " in the packet: " + data,
                    "output":  "<" + api_calls[index] + ">"
                }
            )
        else:
            dataset.append(
                {
                    "instruction": "What is " + knowledge_fields[index] + " in the packet: " + data,
                    "output": "<" + api_calls[index] + ">"
                }
            )

    return dataset
