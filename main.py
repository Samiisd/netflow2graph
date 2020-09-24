#!/usr/bin/env python3

import subprocess
import logging
import sys
import csv
import gzip
import socket

from pathlib import Path

import dpkt
import pyshark

def pcap2csv(p_input: Path, p_output: Path) -> bool:
    args_fields = ["frame.time_epoch", "ip.src", "ip.dst", "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "frame.protocols", "frame.len"]
    args_extras = ["-E separator=,", "-E header=y", "!ipv6"]

    cmd = f"tshark -r {p_input}\
            -T fields {' '.join('-e ' + x for x in args_fields)}\
            {' '.join(args_extras)}"

    logging.debug(f"`pcap2csv:tshark` commande: '{cmd}'")
    with p_output.open("w") as f_out:
        with subprocess.Popen(cmd, stdout=f_out, stderr=subprocess.PIPE, shell=True) as process:
            logging.info(f"`pcap2csv:tshark`: processing {p_input.name}...")
            _, stderr = process.communicate()
            if process.returncode != 0:
                logging.error(f"`pcap2csv:tshark` failed: {stderr.decode('utf-8')}")
                return False
        logging.info(f"`pcap2csv:tshark`: {p_input.name} converted processed into {p_output.name}")
    return True


def dpkt_generator(cap):
    def method():
        for ts, buf in cap: 
            eth = dpkt.ethernet.Ethernet(buf) 
            if not isinstance(eth.data, dpkt.ip.IP): 
              continue 
            ip = eth.data 
            ip_src, ip_dst = inet_to_str(ip.src), inet_to_str(ip.dst) 
            proto = ip.get_proto(ip.p).__name__ 
            if not (isinstance(ip.data, dpkt.tcp.TCP) or isinstance(ip.data, dpkt.udp.UDP)): 
              continue
            yield (ts, ip_src, ip_dst, ip.data.sport, ip.data.dport, ip.len)
    return method

def dump_packets(p_input, csvfile):
    capture = pyshark.FileCapture(str(p_input))

    writer = csv.writer(csvfile)
    writer.writerow(['time', 'ip_src', 'ip_dst', 'port_src', 'port_dst', 'protocol', 'length'])
    packet_ignored = 0

    for packet in capture:
        ip = getattr(packet, 'ip', None)
        tcpudp = getattr(packet, 'tcp', None) or getattr(packet, 'udp', None)
        info = getattr(packet, 'frame_info', None)

        if not (ip and tcpudp and info):
            packet_ignored += 1
            continue

        row = [info.time_epoch, ip.src, ip.dst, tcpudp.srcport, tcpudp.dstport, info.protocols, packet.length]
        if any(map(lambda x: x is None, row)):
            packet_ignored += 1
            continue
        writer.writerow(row)

    return packet_ignored

def inet_to_str(inet): 
    try: 
        return socket.inet_ntop(socket.AF_INET, inet) 
    except ValueError: 
        #return socket.inet_ntop(socket.AF_INET6, inet) 
        return None


def dump_packets2(p_input: Path, csvfile):
    writer = csv.writer(csvfile)
    writer.writerow(['time', 'ip_src', 'ip_dst', 'port_src', 'port_dst', 'protocol', 'length'])
    packet_ignored = 0

    with p_input.open("rb") as fs_pcap:
        cap = dpkt.pcap.Reader(fs_pcap)
        for ts, buf in cap: 
            eth = dpkt.ethernet.Ethernet(buf) 
            if not isinstance(eth.data, dpkt.ip.IP): 
                logging.debug("skipping: no instance of IP")
                packet_ignored += 1
                continue
            ip = eth.data
            ip_src, ip_dst = inet_to_str(ip.src), inet_to_str(ip.dst) 
            if not (ip_src and ip_dst):
                logging.debug(f"skipping: no ip src or dst ({ip_src} => {ip_dst})")
                packet_ignored += 1
                continue
            proto = ip.get_proto(ip.p).__name__ 
            port_src, port_dst = 0, 0
            if isinstance(ip.data, dpkt.tcp.TCP) or isinstance(ip.data, dpkt.udp.UDP):
                port_src, port_dst = ip.data.sport, ip.data.dport
            elif not isinstance(ip.data, dpkt.icmp.ICMP):
                logging.debug(f"skipping: no instance TCP/UDP/ICMP")
                packet_ignored += 1
                continue
            writer.writerow([ts, ip_src, ip_dst, port_src, port_dst, proto, ip.len])

    return packet_ignored

def pcap2csv_light(p_input: Path, p_output: Path) -> bool:

    logging.info(f"pcap2csv:{p_input.name}:starting... (output in {p_output})")
    with gzip.open(p_output, 'wt', newline='') as compressed_csvfile:
    #with p_output.open('w', newline='') as compressed_csvfile:
        packet_ignored = dump_packets2(p_input, compressed_csvfile)

    logging.info(f"pcap2csv:{p_input.name}:done (ignored packet={packet_ignored})")
    return True

def main():
    logging.basicConfig(level=logging.INFO)
    p_in, p_out = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else (sys.argv[1] + '.csv.bz2'))
    if pcap2csv_light(p_in, p_out):
        return 0
    return -1

if __name__ == '__main__':
    sys.exit(main())

