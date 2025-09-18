with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()
    print(f'Total characters: {len(content)}')
    if len(content) > 9309:
        print(f'Character at position 9309: {repr(content[9309])}')
        print(f'Context around 9309: {repr(content[9300:9320])}')
        
        # Look for surrounding context
        start = max(0, 9309-100)
        end = min(len(content), 9309+100)
        print(f'Larger context ({start}-{end}):')
        print(content[start:end])
    else:
        print('Position 9309 is beyond the file length')