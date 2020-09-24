#! /usr/bin/env python3
from scapy.utils import rdpcap, wrpcap, PcapReader, PcapWriter
from scapy.layers.inet import IP, TCP, UDP
from pathlib import Path

from tqdm import tqdm
from zipfile import ZipFile
import os

import sys

def pcap_truncate(path_in, path_out):
    print(f"'{path_in.name}' ({os.path.getsize(str(path_in))/(1024 * 1024)})...")
    cap = rdpcap(str(path_in))
    out = PcapWriter(str(path_out))
    for packet in tqdm(cap, desc="Truncating PCAP", leave=False):
        if packet.haslayer("IP"):
            if packet.haslayer("TCP"):
                packet["IP"]["TCP"].remove_payload()
            elif packet.haslayer("UDP"):
                packet["IP"]["UDP"].remove_payload()
        out.write(packet)
    out.close()

def main(root="./"):
    p_in = Path(sys.argv[1])
    if not (p_in.parent / f"{p_in.name}.pcap").exists():
        pcap_truncate(p_in, p_in.parent / f"{p_in.name}.pcap")
    else:
        os.remove(str(p_in))
    return 0

if __name__ == "__main__":
    sys.exit(main())

