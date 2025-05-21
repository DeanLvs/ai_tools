
# server_kcp_async.py
# 依赖：pip install kcp
import asyncio
import json
from kcp.server import KCPServerAsync, Connection

HOST = "0.0.0.0"
PORT = 8766
TICK_RATE = 20  # Hz
TICK_DT = 1.0 / TICK_RATE
COLLIDE_THRESH = 0.5  # 碰撞半径
HIT_PAUSE = 0.2        # 钩中停顿（秒）

# 全局状态
players = {}  # playerId -> {'x','y','dx','dy','weight'}
hooks   = {}  # playerId -> {'x','y','dx','dy','active','waiting','waitTimer'}
crates = {
    1: {'x': 2.0,  'y': 1.0,  'weight': 20.0},
    2: {'x': -3.0, 'y': 2.5,  'weight': 25.0}
}
next_pid = 1

async def on_connect(conn: Connection):
    """
    新客户端接入，分配 playerId 并发送 assignId
    """
    global next_pid
    pid = next_pid; next_pid += 1
    conn.meta = pid
    players[pid] = {'x':0.0, 'y':0.0, 'dx':0.0, 'dy':0.0, 'weight':30.0}
    hooks[pid]   = {'x':0.0, 'y':0.0, 'dx':0.0, 'dy':0.0,
                    'active':False, 'waiting':False, 'waitTimer':0.0}
    print(f"新玩家 {conn.address} 分配 playerId={pid}")
    await conn.send(json.dumps({'type':'assignId','id':pid}).encode())

async def on_data(conn: Connection, data: bytes):
    """
    处理客户端上报的输入/发钩指令
    """
    pid = conn.meta
    try:
        msg = json.loads(data.decode())
        t = msg.get('type')
        if t == 'input':
            players[pid]['dx'] = msg.get('x',0.0)
            players[pid]['dy'] = msg.get('y',0.0)
        elif t == 'fireHook':
            hooks[pid].update({
                'x': players[pid]['x'], 'y': players[pid]['y'],
                'dx': msg.get('x',0.0), 'dy': msg.get('y',0.0),
                'active':True, 'waiting':False, 'waitTimer':0.0
            })
    except Exception as e:
        print(f"解析客户端数据异常: {e}")

async def game_loop(server: KCPServerAsync):
    """
    主循环：物理更新 + 钩子逻辑 + 碰撞 + 停顿 + 广播
    """
    while True:
        # 玩家移动
        for p in players.values():
            p['x'] += p['dx'] * 5.0 * TICK_DT
            p['y'] += p['dy'] * 5.0 * TICK_DT

        # 钩子更新
        for pid, h in hooks.items():
            if not h['active']:
                continue
            # 钩中停顿处理
            if h['waiting']:
                h['waitTimer'] += TICK_DT
                if h['waitTimer'] >= HIT_PAUSE:
                    h['active'] = False
                    h['waiting'] = False
                continue

            # 钩子飞行
            h['x'] += h['dx'] * 10.0 * TICK_DT
            h['y'] += h['dy'] * 10.0 * TICK_DT

            # 碰撞检测
            collided = False
            # 与玩家
            for opid, p2 in players.items():
                if opid == pid: continue
                dx = h['x'] - p2['x']; dy = h['y'] - p2['y']
                if dx*dx + dy*dy < COLLIDE_THRESH*COLLIDE_THRESH:
                    # 按重量决定谁跑
                    if players[pid]['weight'] < p2['weight']:
                        players[pid]['x'], players[pid]['y'] = p2['x'], p2['y']
                    else:
                        players[opid]['x'], players[opid]['y'] = players[pid]['x'], players[pid]['y']
                    collided = True
                    break
            # 与箱子
            if not collided:
                for cid, c in crates.items():
                    dx = h['x'] - c['x']; dy = h['y'] - c['y']
                    if dx*dx + dy*dy < COLLIDE_THRESH*COLLIDE_THRESH:
                        if players[pid]['weight'] < c['weight']:
                            players[pid]['x'], players[pid]['y'] = c['x'], c['y']
                        else:
                            c['x'], c['y'] = players[pid]['x'], players[pid]['y']
                        collided = True
                        break
            # 如果碰撞，进入停顿
            if collided:
                h['waiting'] = True
                h['waitTimer'] = 0.0

        # 广播最新状态
        state = {
            'type':'state',
            'players': [{'id':pid, 'x':p['x'], 'y':p['y']} for pid,p in players.items()],
            'hooks':   [{'ownerId':pid, 'x':h['x'], 'y':h['y'], 'active':h['active']} for pid,h in hooks.items()],
            'crates':  [{'id':cid, 'x':c['x'], 'y':c['y']} for cid,c in crates.items()]
        }
        blob = json.dumps(state).encode()
        for conn in list(server._connections):
            await conn.send(blob)

        await asyncio.sleep(TICK_DT)

async def main():
    # 初始化 KCP 异步服务器
    server = KCPServerAsync(HOST, PORT, conv_id=123, no_delay=True)
    server.on_start   = lambda: print(f"KCP 服务器启动 {HOST}:{PORT}, {TICK_RATE}Hz")
    server.on_connect = on_connect
    server.on_data    = on_data

    # 并发运行监听和游戏逻辑
    await asyncio.gather(
        server.listen(),
        game_loop(server)
    )

if __name__ == '__main__':
    asyncio.run(main())

