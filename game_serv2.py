import asyncio
import json

HOST = '0.0.0.0'
PORT = 8766
TICK_RATE = 20
dt = 1.0 / TICK_RATE

players = {}
hooks = {}
next_id = 1
clients = {}

async def handle_client(reader, writer):
    global next_id
    addr = writer.get_extra_info('peername')
    clients[addr] = writer

    pid = next_id
    next_id += 1
    players[pid] = {'x': 0.0, 'y': 0.0, 'dx': 0.0, 'dy': 0.0, 'weight': 30}
    hooks[pid] = {'x': 0.0, 'y': 0.0, 'dx': 0.0, 'dy': 0.0, 'active': False, 'waiting': False, 'timer': 0}

    # 发送分配的 playerId
    assign = json.dumps({'type': 'assignId', 'id': pid}).encode()
    writer.write(assign + b'\n')
    await writer.drain()

    while True:
        try:
            data = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not data:
                break
            msg = json.loads(data.decode())
            if msg['type'] == 'input':
                players[msg['id']]['dx'] = msg['x']
                players[msg['id']]['dy'] = msg['y']
            elif msg['type'] == 'fireHook':
                hooks[msg['id']].update({
                    'x': players[msg['id']]['x'],
                    'y': players[msg['id']]['y'],
                    'dx': msg['x'],
                    'dy': msg['y'],
                    'active': True,
                    'waiting': False,
                    'timer': 0
                })
        except asyncio.TimeoutError:
            continue

async def game_loop():
    while True:
        # 更新玩家位置
        for p in players.values():
            p['x'] += p['dx'] * 5 * dt
            p['y'] += p['dy'] * 5 * dt

        # 更新钩子
        for pid, h in hooks.items():
            if not h['active']:
                continue
            if h['waiting']:
                h['timer'] += dt
                if h['timer'] >= 0.2:
                    h['active'] = False
                    h['waiting'] = False
                continue

            h['x'] += h['dx'] * 10 * dt
            h['y'] += h['dy'] * 10 * dt

            # 碰撞检测
            for opid, p2 in players.items():
                if opid == pid:
                    continue
                dx = h['x'] - p2['x']
                dy = h['y'] - p2['y']
                if dx * dx + dy * dy < 0.5:
                    # 重量比较
                    if players[pid]['weight'] < p2['weight']:
                        players[pid]['x'], players[pid]['y'] = p2['x'], p2['y']
                    else:
                        p2['x'], p2['y'] = players[pid]['x'], players[pid]['y']
                    h['waiting'] = True
                    h['timer'] = 0
                    break

        # 广播场景状态
        state = {
            'type': 'state',
            'players': [{'id': pid, 'x': p['x'], 'y': p['y']} for pid, p in players.items()],
            'hooks': [{'ownerId': pid, 'x': h['x'], 'y': h['y'], 'active': h['active']} for pid, h in hooks.items()]
        }
        blob = (json.dumps(state) + '\n').encode()
        for w in list(clients.values()):
            try:
                w.write(blob)
                await w.drain()
            except:
                continue

        await asyncio.sleep(dt)

async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    print(f"服务器已启动 {HOST}:{PORT}")
    await asyncio.gather(server.serve_forever(), game_loop())

if __name__ == '__main__':
    asyncio.run(main())