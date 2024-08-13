import re
import csv
from collections import defaultdict
import os
import pandas as pd
import csv
import re
import os
from collections import Counter
from tqdm import tqdm
from fuzzywuzzy import fuzz
import argparse

md5_regex = r'\b([a-fA-F\d]{32})\b'  # MD5 hashes
sha1_regex = r'\b([a-fA-F\d]{40})\b'  # SHA1 hashes
sha256_regex = r'\b([a-fA-F\d]{64})\b'  # SHA256 hashes
ipv4_regex = r'\b(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\b'  # IPv4 addresses
ipv6_regex = r'\b((([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4})|(([0-9a-fA-F]{1,4}:){1,7}:)|(([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4})|(([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2})|(([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3})|(([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4})|(([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5})|([0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:))|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))\b'  # IPv6 addresses
c2_regex = r'\bC2:\s*(.+)\b'  # Command and Control information
uri_regex = r'\bURI:\s*(.+)\b'  # URIs
user_agent_regex = r'\bUser-Agent:\s*(.+)\b'  # User agents
smtp_mailer_regex = r'\bX-Mailer:\s*(.+)\b'  # SMTP mailer
mac_address_regex = r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'  # MAC addresses
email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email addresses
cipher_suite_regex = r'\b(TLS_(?:DHE|RSA|ECDHE)_WITH_AES_\d{3}_GCM_SHA\d{2})\b'  # Cipher suites
asn_regex = r'\bAS\d+\b'  # Autonomous System Numbers
subnet_mask_regex = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})/\d{1,2}\b'  # Subnet masks
tls_version_regex = r'\b(TLSv1\.2|TLSv1\.3)\b'  # TLS versions
cookie_regex = r'\bSet-Cookie:\s*(.+)\b'  # Cookies
ssh_fingerprint_regex = r'\b([0-9a-f]{2}:){15}[0-9a-f]{2}\b'  # SSH fingerprints
protocol_name_regex = r'\b(SSL|DNS|FTP|SMTP)\b'  # Protocol names
snmp_trap_regex = r'\bSNMP Trap:\s*(.+)\b'  # SNMP traps
http_header_regex = r'\b(Accept|Content-Type|Cache-Control):\s*(.+)\b'  # HTTP headers
arp_entry_regex = r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\s+(\d{1,3}\.){3}\d{1,3}\b'  # ARP entries
service_name_regex = r'\b(ssh|ftp|dns)\b'  # Service names
snmp_mib_regex = r'\b1\.3\.6\.1\.\d+\.\d+\b'  # SNMP MIB objects
device_name_regex = r'\b(router|switch|firewall)\b'  # Device names
ssl_certificate_regex = r'\bSubject: CN=(.*?)(?:,|$)\b'  # SSL certificate details
routing_protocol_regex = r'\b(OSPF|BGP|RIP)\b'  # Routing protocols
packet_capture_timestamp_regex = r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\b'  # Packet capture timestamps
traffic_volume_regex = r'\b(\d+)\s*(bytes|packets)\b'  # Traffic volume
protocol_field_regex = r'\b(IP|TCP|UDP)\.(.+)\b'  # Protocol fields
virtualization_tech_regex = r'\b(VLAN|VXLAN)\b'  # Virtualization technologies
security_protocol_regex = r'\b(IPSec|SSL/TLS|WPA2)\b'  # Security protocols
tunneling_protocol_regex = r'\b(GRE|L2TP|PPTP)\b'  # Tunneling protocols
auth_method_regex = r'\b(EAP|Kerberos|RADIUS)\b'  # Authentication methods
file_transfer_protocol_regex = r'\b(FTP|SFTP|SCP)\b'  # File transfer protocols
transport_security_regex = r'\b(DTLS|TLS 1\.3)\b'  # Transport security protocols
nat_info_regex = r'\b(Source NAT|Destination NAT)\b'  # NAT information
proxy_server_regex = r'\b(Forward Proxy|Reverse Proxy|Proxy|Server)\b'  # Proxy servers


regex_list_head = [
    ("md5_regex", md5_regex),
    ("sha1_regex", sha1_regex),
    ("sha256_regex", sha256_regex),
    ("ipv4_regex", ipv4_regex),
    ("ipv6_regex", ipv6_regex),
    ("c2_regex", c2_regex),
    ("uri_regex", uri_regex),
    ("user_agent_regex", user_agent_regex),
    ("smtp_mailer_regex", smtp_mailer_regex),
    ("mac_address_regex", mac_address_regex),
    ("email_regex", email_regex),
    ("cipher_suite_regex", cipher_suite_regex),
    ("asn_regex", asn_regex),
    ("subnet_mask_regex", subnet_mask_regex),
    ("tls_version_regex", tls_version_regex),
    ("cookie_regex", cookie_regex),
    ("ssh_fingerprint_regex", ssh_fingerprint_regex),
    ("protocol_name_regex", protocol_name_regex),
    ("snmp_trap_regex", snmp_trap_regex),
    ("http_header_regex", http_header_regex),
    ("arp_entry_regex", arp_entry_regex),
    ("service_name_regex", service_name_regex),
    ("snmp_mib_regex", snmp_mib_regex),
    ("device_name_regex", device_name_regex),
    ("ssl_certificate_regex", ssl_certificate_regex),
    ("routing_protocol_regex", routing_protocol_regex),
    ("packet_capture_timestamp_regex", packet_capture_timestamp_regex),
    ("traffic_volume_regex", traffic_volume_regex),
    ("protocol_field_regex", protocol_field_regex),
    ("virtualization_tech_regex", virtualization_tech_regex),
    ("security_protocol_regex", security_protocol_regex),
    ("tunneling_protocol_regex", tunneling_protocol_regex),
    ("auth_method_regex", auth_method_regex),
    ("file_transfer_protocol_regex", file_transfer_protocol_regex),
    ("transport_security_regex", transport_security_regex),
    ("nat_info_regex", nat_info_regex),
    ("proxy_server_regex", proxy_server_regex)
]

non_technical = ['0day', 'C2', 'IDS', 'IP Security', 'IPS', 'account', 'accounts', 'adware', 'amplification attack', 'antivirus', 'attack', 'backdoor', 'ban', 'banned', 
                 'banning', 'black hat', 'block', 'blocked', 'blocking', 'bot', 'bots', 'bot herded', 'bot herder', 'bot herding', 'botmaster', 'botnet', 
                 'breach', 'breached', 'breaching', 'break into', 'browser hijacker', 'browser hijacking', 'bypass', 'bypassed', 'bypassing', 'cheat', 'cheated', 
                 'cheating', 'ciphertext', 'clickjack', 'clickjacked', 'clickjacking', 'cloned cards', 'code', 'coded', 'coding', 'command and control server', 
                 'compromised', 'compromise', 'compromising', 'confidential', 'crack', 'cracked', 'cracking', 'crash', 'crashed', 'crashing', 'crimeware', 
                 'crypter', 'crypto-locked', 'crypto-locker', 'crypto-locking', 'cryptography', 'cryptology', 'cyber bullying', 'cyber conflict', 
                 'cyber espionage', 'cyber harassment', 'cyber risk', 'cyber spy', 'cyber terrorism', 'data breach', 'data leakage', 'data loss', 
                 'data theft', 'database', 'ddos', 'deactivate', 'deactivated', 'deactivating', 'denial of service', 'digital sabotage', 'downloader', 
                 'doxing', 'doxxing', 'doxxers', 'dropper', 'dump', 'dumps', 'encrypt', 'encrypted', 'encrypting', 'encryption', 'error', 'evade', 
                 'evaded', 'evading', 'evasion', 'exfiltrate', 'exfiltrated', 'exfiltrating', 'exfiltration', 'exploit', 'exploits', 'exploit kit', 'exploited', 
                 'exploiting', 'fake', 'faking', 'fast flux', 'file', 'firewall', 'geoblock', 'geoblocked', 'geoblocker', 'geoblocking', 'gift card', 
                 'hack', 'hacked', 'hackers', 'hacking', 'hacktivism', 'hacktivist', 'hijack', 'hijacked', 'hijacking', 'honeypot', 'identity theft', 
                 'infect', 'infected', 'infecting', 'inject', 'injected', 'injecting', 'injection', 'invade', 'invading', 'invaded', 'keylogger', 'leach', 
                 'leacher', 'leaching', 'leak', 'leaks', 'leaked', 'leaking', 'licensed', 'macro', 'malicious', 'malicious code', 'maliciously', 
                 'malvertised scareware', 'malvertising', 'malware', 'obfuscate', 'obfuscated', 'obfuscation', 'offset', 'offsets', 'outdated', 'packet sniffing', 
                 'password', 'patch', 'patched', 'patching', 'payload', 'pen test', 'pen tested', 'pen tester', 'pen testing', 'pentest', 'pentested', 
                 'pentester', 'pentesting', 'personal', 'pharming', 'phished', 'phisher', 'phishing', 'prepaid card', 'prepaid cards', 'private key', 
                 'proxy', 'proxied', 'ransomware', 'rat', 'reflection attack', 'reverse', 'root to local attack', 'rootkit', 'scam', 'scamming', 'scammed', 
                 'scams', 'scan', 'scans', 'scanned', 'scanner', 'scanning', 'script', 'script kiddies', 'scripted', 'scripting', 'security', 
                 'security breach', 'security incident', 'security threat', 'sensitive', 'server', 'servers', 'session', 'sessions', 'shell', 'sniff', 'sniffed', 'sniffer', 'sniffing', 
                 'spam', 'spammed', 'spammer', 'spamming', 'spearphising', 'spoof', 'spoofs', 'spoofed', 'spoofer', 'spoofing', 'spyware', 'spy', 
                 'spying', 'sql', 'stealware', 'steal', 'stealth', 'stolen', 'stress', 'stresser', 'tcp', 'threat hunters', 'topper', 'torrent', 'trojan', 
                 'twishing', 'unlicensed', 'unpatched', 'user to root attack', 'virus', 'vishing', 'vpn', 'vpns', 'vulnerability', 'white hat', 'worm', 
                 'zero-day']

def combined_analysis_fuzzy(data_file, data_column, regex_list, software_file, software_column, non_technical, output_file, fuzzy_matching_cutoff=80, encoding='utf-8', field_size_limit=10000000):
    """
    Analyzes the specified column in the CSV file against a list of regex patterns, MITRE software names,
    and non-technical keywords using both exact and fuzzy matching with a threshold of 80%, and outputs a new CSV file with additional columns for each match.

    Parameters:
    - data_file (str): The name of the input CSV file.
    - data_column (str): The name of the column to analyze.
    - regex_list (list): A list of tuples where each tuple contains a name and a regex pattern.
    - software_file (str): The path to the software Excel file.
    - software_column (str): The column name in the software Excel file to use for matching.
    - non_technical (list): A list of non-technical keywords to match.
    - output_file (str): The name of the output CSV file.
    - encoding (str, optional): The file encoding. Default is 'utf-8'.
    - field_size_limit (int, optional): The maximum field size limit. Default is 1000000.
    
    Returns:
    - tuple: A tuple containing three dictionaries:
        1. Counts of matches for each regex pattern (excluding those with 0 hits).
        2. Counts of matches for each MITRE software.
        3. Counts of matches for each non-technical keyword.
    """
    # Increase the CSV field size limit
    csv.field_size_limit(field_size_limit)
    
    # Create dictionaries to store counts
    regex_counts = Counter()
    mitre_counts = Counter()
    non_technical_counts = Counter()

    # Load MITRE software names
    software_df = pd.read_excel(software_file, sheet_name='software')
    mitre_names = software_df[software_column].tolist()

    # Create the output folder if it doesn't exist
    output_folder = os.path.dirname(output_file)
    os.makedirs(output_folder, exist_ok=True)

    try:
        with open(data_file, 'r', encoding=encoding) as infile, \
             open(output_file, 'w', newline='', encoding=encoding) as outfile:
            
            csv_reader = csv.DictReader(infile)
            headers = csv_reader.fieldnames

            if data_column not in headers:
                raise ValueError(f"Column '{data_column}' not found in CSV headers: {headers}")

            # Remove 'html_code' from headers if it exists
            output_headers = [header for header in headers if header != 'html_code']
            
            # Add new columns
            new_columns = ['regex', 'mitre_software', 'non_technical', 'technical?', 'regex_fuzzy_ratio', 'mitre_fuzzy_ratio', 'non_technical_fuzzy_ratio']
            output_headers.extend(new_columns)
            
            csv_writer = csv.DictWriter(outfile, fieldnames=output_headers)
            csv_writer.writeheader()

            # Calculate the nr of rows
            total_rows = sum(1 for _ in csv_reader)
            infile.seek(0)
            next(csv_reader)  # Skip the header row
            pbar = tqdm(total=total_rows, desc="Processing rows")

            for row in csv_reader:
                content = row[data_column]
                base_row = {key: value for key, value in row.items() if key != 'html_code'}

                regex_matches = []
                mitre_matches = []
                non_technical_matches = []

                regex_fuzzy_ratios = []
                mitre_fuzzy_ratios = []
                non_technical_fuzzy_ratios = []

                # Match regular expressions using exact and fuzzy matching
                for name, pattern in regex_list:
                    if re.search(pattern, content):
                        regex_counts[name] += 1
                        regex_matches.append(name)
                        regex_fuzzy_ratios.append('100')  # Exact match
                    elif len(pattern) > 5:  # Only apply fuzzy matching if the word is longer than 5 characters
                        regex_ratio = fuzz.partial_ratio(content, pattern)
                        if regex_ratio >= 80 and re.search(re.escape(pattern) + r'\w', content):  # Match only if there's a subsequent character
                            regex_counts[name] += 1
                            regex_matches.append(name)
                            regex_fuzzy_ratios.append(str(regex_ratio))

                # Match MITRE software names using exact matching only
                for name in mitre_names:
                    if re.search(r'\b' + re.escape(name) + r'\b', content, re.IGNORECASE):
                        mitre_counts[name] += 1
                        mitre_matches.append(name)
                        mitre_fuzzy_ratios.append('100')  # Exact match

                # Match non-technical keywords using exact and fuzzy matching
                for keyword in non_technical:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', content, re.IGNORECASE):
                        non_technical_counts[keyword] += 1
                        non_technical_matches.append(keyword)
                        non_technical_fuzzy_ratios.append('100')  # Exact match
                    elif len(keyword) > 5:  # Only apply fuzzy matching if the word is longer than 5 characters
                        non_technical_ratio = fuzz.partial_ratio(content, keyword)
                        if non_technical_ratio >= 80 and re.search(re.escape(keyword) + r'\w', content):  # Match only if there's a subsequent character
                            non_technical_counts[keyword] += 1
                            non_technical_matches.append(keyword)
                            non_technical_fuzzy_ratios.append(str(non_technical_ratio))

                # Only write rows with at least one match
                if regex_matches or mitre_matches or non_technical_matches:
                    # Determine the type of content
                    if (regex_matches or mitre_matches) and non_technical_matches:
                        technical_type = 'both'
                    elif regex_matches or mitre_matches:
                        technical_type = 'technical'
                    elif non_technical_matches:
                        technical_type = 'non-technical'
                    else:
                        technical_type = ''

                    # Combine matches into single row
                    new_row = base_row.copy()
                    new_row.update({
                        'regex': ', '.join(regex_matches),
                        'mitre_software': ', '.join(mitre_matches),
                        'non_technical': ', '.join(non_technical_matches),
                        'technical?': technical_type,
                        'regex_fuzzy_ratio': ', '.join(f'{ratio}%' for ratio in regex_fuzzy_ratios),
                        'mitre_fuzzy_ratio': ', '.join(f'{ratio}%' for ratio in mitre_fuzzy_ratios),
                        'non_technical_fuzzy_ratio': ', '.join(f'{ratio}%' for ratio in non_technical_fuzzy_ratios)
                    })
                    csv_writer.writerow(new_row)
                # Update progress bar
                pbar.update(1)

            pbar.close()

        print(f"Output CSV file '{output_file}' has been created.")
        
        # Filter out regex patterns with 0 hits
        non_zero_regex_counts = {name: count for name, count in regex_counts.items() if count > 0}
        
        return dict(non_zero_regex_counts), dict(mitre_counts), dict(non_technical_counts)
        
    except FileNotFoundError as e:
        print(f"File '{data_file}' not found.")
        print(f"Error message: {e}")
        return None, None, None
    except UnicodeDecodeError as e:
        print(f"Error decoding file '{data_file}': {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Dictionary Mapping Script')
    parser.add_argument('--inputfile', required=True, help='Path to the input CSV file')
    parser.add_argument('--input_column', required=True, help='Name of the column to analyze')
    parser.add_argument('--software_location', required=True, help='Path to the software Excel file')
    parser.add_argument('--software_name', required=True, help='Column name in the software Excel file to use for matching')
    parser.add_argument('--outputfilepath', required=True, help='Path to the output CSV file')
    parser.add_argument('--fuzzy_cutoff', type=float, required=True, help='Percentage above which we keep the match')

    args = parser.parse_args()

    regex_list = regex_list_head
    non_technical_list = non_technical 

    combined_analysis_fuzzy(
        data_file=args.inputfile,
        data_column=args.input_column,
        regex_list=regex_list,
        software_file=args.software_location,
        software_column=args.software_name,
        non_technical=non_technical_list,
        output_file=args.outputfilepath,
        fuzzy_matching_cutoff=args.fuzzy_cutoff
    )


if __name__ == '__main__':
    main()
