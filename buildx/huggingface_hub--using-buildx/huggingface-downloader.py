#!/usr/bin/env python3
# https://huggingface.co/docs/huggingface_hub/guides/download#download-an-entire-repository
import os
import argparse

from huggingface_hub import snapshot_download, hf_hub_download, login

parser = argparse.ArgumentParser()

parser.add_argument('repos', type=str, nargs='+', default=[], help="HuggingFace model, datasets, or file names to download")

parser.add_argument('--type', type=str, default='model', choices=['model', 'dataset'], help="'model' or 'dataset'")
parser.add_argument('--token', type=str, default=os.environ.get('HUGGINGFACE_TOKEN', ''), help="HuggingFace account login token from https://huggingface.co/docs/hub/security-tokens (defaults to $HUGGINGFACE_TOKEN)")
parser.add_argument('--cache-dir', type=str, default=os.environ.get('TRANSFORMERS_CACHE', '/root/.cache/huggingface'), help="Location to download the repo to (defaults to $TRANSFORMERS_CACHE)")
parser.add_argument('--location-file', type=str, default='/tmp/hf_download', help="file to write the local location/path of the downloaded repo(s) to")

parser.add_argument('--allow-patterns', type=str, default='', help="comma-separated list of file patterns to download (enclose in single quotes if using wildcards)")
parser.add_argument('--ignore-patterns', type=str, default='', help="comma-separated list of file patterns to exclude from downloading (enclose in single quotes if using wildcards)")
parser.add_argument('--skip-safetensors', action='store_true', help="filter out the downloading of .safetensor files")

args = parser.parse_args()

args.allow_patterns = [x for x in args.allow_patterns.split(',') if x]
args.ignore_patterns = [x for x in args.ignore_patterns.split(',') if x]

if args.skip_safetensors:
    args.ignore_patterns.append('*.safetensors')
  
if len(args.allow_patterns) == 0:
    args.allow_patterns = None
    
if len(args.ignore_patterns) == 0:
    args.ignore_patterns = None

if args.token:
    print("Logging into HuggingFace Hub...")
    login(token=args.token)
    
locations = []

for repo in args.repos:
    if os.path.isdir(repo) or os.path.isfile(repo):
        print(f"\nPath to local directory or file given: {repo}")
        locations.append(repo)
        continue
        
    print(f"\nDownloading {repo} to {args.cache_dir}\n")
    
    # handle either "org/repo" or individual "org/repo/file"
    # the former has 0-1 slashes, while the later has 2.
    num_slashes = 0
    
    for c in repo:
        if c == '/':
            num_slashes += 1
            
    if num_slashes >= 2:  
        slash_count = 0
        
        for idx, i in enumerate(repo):
            if i == '/':
                slash_count += 1
                if slash_count == 2:
                    break
                    
        repo_id = repo[:idx]
        filename = repo[idx+1:]

        print('repo_id ', repo_id)
        print('filename', filename)
        
        repo_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=args.type, cache_dir=args.cache_dir, resume_download=True)
    else:
        repo_path = snapshot_download(repo_id=repo, repo_type=args.type, cache_dir=args.cache_dir, resume_download=True, 
                                      allow_patterns=args.allow_patterns, ignore_patterns=args.ignore_patterns)
        
    locations.append(repo_path)
    print(f"\nDownloaded {repo} to: {repo_path}")
    
if args.location_file:
    with open(args.location_file, 'w') as file:
        file.write('\n'.join(locations))
