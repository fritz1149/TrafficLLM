from flowcontainer.extractor import extract
import binascii
import scapy.all as scapy
import os


# MAX_PACKET_NUMBER = 10  # 10
MAX_PACKET_NUMBER = 5
MAX_PACKET_LENGTH_IN_FLOW = 256
HEX_PACKET_START_INDEX = 0  # 48 # 76


def build_flow_data(pcap_file, flow_feature="flow bytes", max_packet_number=MAX_PACKET_NUMBER):

    build_data = []

    if flow_feature == "flow bytes":

        # flow bytes feature
        packets = scapy.rdpcap(pcap_file)

        hex_stream = []
        for i, packet in enumerate(packets):
            if i >= max_packet_number:
                break
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))

            packet_string = data.decode()

            # byte_list = re.findall(".{2}", packet_string)
            # packet_string = " ".join(byte_list)

            hex_stream.append(packet_string[HEX_PACKET_START_INDEX:min(len(packet_string), MAX_PACKET_LENGTH_IN_FLOW)])

        flow_data = "<pck>" + "<pck>".join(hex_stream)
        build_data.append(flow_data)

    elif flow_feature == "flow sequence":
        flows = extract(pcap_file,
                        filter='tcp or udp',
                        extension=["tcp.payload", "udp.payload"],
                        split_flag=False,
                        verbose=True)

        # flow sequence feature
        for key, flow in flows.items():
            flow_seq = []

            length_seq = flow.lengths
            for i, packet_length in enumerate(length_seq):
                if i >= max_packet_number:
                    break
                flow_seq.append(str(packet_length))

            flow_data = " ".join(flow_seq)
            build_data.append(flow_data)

    elif flow_feature == "traffic words":
        tmp_path = f"../tmp/tmp-{os.getpid()}.txt"

        # tshark 3.6.16
        # fields = ["frame.encap_type", "frame.time", "frame.offset_shift", "frame.time_epoch", "frame.time_delta",
        #           "frame.time_relative", "frame.number", "frame.len", "frame.marked", "frame.protocols", "eth.dst",
        #           "eth.dst_resolved", "eth.dst.oui", "eth.dst.oui_resolved", "eth.dst.lg", "eth.dst.ig", "eth.src",
        #           "eth.src_resolved", "eth.src.oui", "eth.src.oui_resolved", "eth.src.lg", "eth.src.ig", "eth.type",
        #           "ip.version", "ip.hdr_len", "ip.dsfield", "ip.dsfield.dscp", "ip.dsfield.ecn", "ip.len", "ip.id",
        #           "ip.flags", "ip.flags.rb", "ip.flags.df", "ip.flags.mf", "ip.frag_offset", "ip.ttl", "ip.proto",
        #           "ip.checksum", "ip.checksum.status", "ip.src", "ip.dst", "tcp.srcport", "tcp.dstport", "tcp.stream",
        #           "tcp.completeness", "tcp.len", "tcp.seq", "tcp.nxtseq", "tcp.ack", "tcp.hdr_len", "tcp.flags",
        #           "tcp.flags.res", "tcp.flags.ns", "tcp.flags.cwr", "tcp.flags.ecn", "tcp.flags.urg", "tcp.flags.ack",
        #           "tcp.flags.push", "tcp.flags.reset", "tcp.flags.syn", "tcp.flags.fin", "tcp.flags.str", "tcp.window_size",
        #           "tcp.window_size_scalefactor", "tcp.checksum", "tcp.checksum.status", "tcp.urgent_pointer", "tcp.time_relative",
        #           "tcp.time_delta", "tcp.analysis.bytes_in_flight", "tcp.analysis.push_bytes_sent", "tcp.segment", "tcp.segment.count",
        #           "tcp.reassembled.length", "tls.record.content_type", "tls.record.version", "tls.record.length", "tcp.payload"]

        # tshark 2.6.10
        fields = ["frame.encap_type", "frame.time", "frame.offset_shift", "frame.time_epoch", "frame.time_delta",
                  "frame.time_relative", "frame.number", "frame.len", "frame.marked", "frame.protocols", 
                #   "eth.dst", "eth.dst_resolved", "eth.src", "eth.src_resolved", "eth.type",
                  "ip.version", "ip.hdr_len", "ip.dsfield", "ip.dsfield.dscp", "ip.dsfield.ecn", "ip.len", "ip.id",
                  "ip.flags", "ip.flags.rb", "ip.flags.df", "ip.flags.mf", "ip.frag_offset", "ip.ttl", "ip.proto",
                  "ip.checksum", "ip.checksum.status", "ip.src", "ip.dst", "tcp.srcport", "tcp.dstport", "tcp.stream",
                  "tcp.len", "tcp.seq", "tcp.nxtseq", "tcp.ack", "tcp.hdr_len", "tcp.flags",
                  "tcp.flags.res", "tcp.flags.cwr", "tcp.flags.urg", "tcp.flags.ack",
                  "tcp.flags.push", "tcp.flags.reset", "tcp.flags.syn", "tcp.flags.fin", "tcp.flags.str",
                  "tcp.window_size", "tcp.window_size_scalefactor", "tcp.checksum", "tcp.checksum.status", "tcp.urgent_pointer",
                  "tcp.time_relative", "tcp.time_delta", "tcp.analysis.bytes_in_flight", "tcp.analysis.push_bytes_sent", "tcp.segment",
                  "tcp.segment.count", "tcp.reassembled.length", "tcp.payload", "udp.srcport", "udp.dstport", "udp.length",
                  "udp.checksum", "udp.checksum.status", "udp.stream", "data.len"]

        extract_str = " -e " + " -e ".join(fields) + " "
        cmd = "tshark -r " + pcap_file + extract_str + "-T fields -Y 'tcp or udp' > " + tmp_path
        os.system(cmd)

        packets = []

        with open(tmp_path, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
        for line in lines[:max_packet_number]:
            values = line[:-1].split("\t")
            if not values or values[0] == "":
                continue

            packet_data = []
            for field, value in zip(fields, values):
                if field == "tcp.flags.str":
                    value = value.encode("unicode_escape").decode("unicode_escape")
                if field == "tcp.payload":
                    value = value[:1000] if len(value) > 1000 else value
                if value == "":
                    continue
                packet_data.append(field + ": " + value)
            packet_data = ", ".join(packet_data)

            packets.append(packet_data)

        if len(packets) > 0:
            flow_data = "<pck>" + "<pck>".join(packets)
            build_data.append(flow_data)

    else:
        # payload bytes feature
        flows = extract(pcap_file,
                        filter='tcp or udp',
                        extension=["tcp.payload", "udp.payload"],
                        split_flag=False,
                        verbose=True)

        for key, flow in flows.items():
            if len(flow.extension.values()) == 0:
                continue
            packet_list = list(flow.extension.values())[0]
            hex_stream = []
            for i, packet in enumerate(packet_list):
                if i >= MAX_PACKET_NUMBER:
                    break
                hex_stream.append(packet[0][:min(len(packet[0]), MAX_PACKET_LENGTH_IN_FLOW)])
            flow_data = "<pck>" + "<pck>".join(hex_stream)
            # print(flow_data)
            build_data.append(flow_data)

    return build_data
