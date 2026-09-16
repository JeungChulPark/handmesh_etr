using System;
using System.Collections.Generic;
using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Text;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// LAN auto-discovery of the PC inference server (infer_ipad_stream.py), so the app
    /// keeps working when DHCP moves the PC to a new address — no rebuild, no fixed IP.
    ///
    /// Protocol (server side: <c>start_discovery_responder</c> in infer_ipad_stream.py):
    ///   app  -> udp:9777  "HANDMESH_DISCOVER?"
    ///   server -> app     "HANDMESH_SERVER &lt;tcp_port&gt;"      (unicast reply to the probe's source)
    /// The reply's SOURCE ADDRESS is the server's IP; the payload carries its TCP port.
    ///
    /// iOS note: broadcast/multicast needs the com.apple.developer.networking.multicast
    /// entitlement, so the primary strategy is a UNICAST SWEEP — one small probe to every
    /// host of each local /24 (≤254 datagrams, a few ms on WiFi). A subnet-broadcast probe
    /// is also attempted opportunistically for networks where it happens to work; failures
    /// are ignored. Local-network permission (the standard iOS prompt) is still required,
    /// but the TCP stream already needs that anyway.
    ///
    /// Blocking — call from a background thread (RgbdStreamer's worker does).
    /// </summary>
    public static class ServerDiscovery
    {
        public const int DefaultDiscoveryPort = 9777;
        const string Probe = "HANDMESH_DISCOVER?";
        const string ReplyPrefix = "HANDMESH_SERVER";

        /// <summary>Found server endpoint.</summary>
        public readonly struct Result
        {
            public readonly string Host;
            public readonly int TcpPort;
            public Result(string host, int tcpPort) { Host = host; TcpPort = tcpPort; }
        }

        /// <summary>
        /// Probe the local network and return the first server that answers, or null.
        /// <paramref name="timeoutMs"/> is the total budget (sweep + listen).
        /// </summary>
        public static Result? FindServer(int discoveryPort = DefaultDiscoveryPort, int timeoutMs = 1500)
        {
            byte[] probe = Encoding.ASCII.GetBytes(Probe);
            try
            {
                using var udp = new UdpClient(0);            // ephemeral port, replies come back here
                udp.Client.SendTimeout = 200;
                try { udp.EnableBroadcast = true; } catch { /* iOS without entitlement */ }

                foreach (IPAddress local in LocalIPv4Addresses())
                {
                    byte[] b = local.GetAddressBytes();
                    // opportunistic subnet broadcast (fast path where allowed)
                    TrySend(udp, probe, new IPEndPoint(
                        new IPAddress(new byte[] { b[0], b[1], b[2], 255 }), discoveryPort));
                    // unicast sweep of the /24 — works everywhere, no entitlement
                    for (int h = 1; h <= 254; h++)
                    {
                        if (h == b[3]) continue;             // skip ourselves
                        TrySend(udp, probe, new IPEndPoint(
                            new IPAddress(new byte[] { b[0], b[1], b[2], (byte)h }), discoveryPort));
                    }
                }

                // collect the first valid reply within the remaining budget
                int deadline = Environment.TickCount + timeoutMs;
                var any = new IPEndPoint(IPAddress.Any, 0);
                while (unchecked(deadline - Environment.TickCount) > 0)
                {
                    udp.Client.ReceiveTimeout = Math.Max(50, unchecked(deadline - Environment.TickCount));
                    byte[] data;
                    var from = any;
                    try { data = udp.Receive(ref from); }
                    catch (SocketException) { break; }       // timed out
                    string msg = Encoding.ASCII.GetString(data).Trim();
                    if (!msg.StartsWith(ReplyPrefix, StringComparison.Ordinal)) continue;
                    int port = 0;
                    int sp = msg.IndexOf(' ');
                    if (sp > 0) int.TryParse(msg.Substring(sp + 1), out port);
                    return new Result(from.Address.ToString(), port);
                }
            }
            catch (Exception)
            {
                // no network / sockets unavailable — caller just retries later
            }
            return null;
        }

        static void TrySend(UdpClient udp, byte[] payload, IPEndPoint to)
        {
            try { udp.Send(payload, payload.Length, to); } catch { /* unreachable host etc. */ }
        }

        static IEnumerable<IPAddress> LocalIPv4Addresses()
        {
            var seen = new HashSet<string>();
            NetworkInterface[] nics;
            try { nics = NetworkInterface.GetAllNetworkInterfaces(); }
            catch (Exception) { yield break; }
            foreach (var nic in nics)
            {
                if (nic.OperationalStatus != OperationalStatus.Up) continue;
                if (nic.NetworkInterfaceType == NetworkInterfaceType.Loopback) continue;
                IPInterfaceProperties props;
                try { props = nic.GetIPProperties(); } catch (Exception) { continue; }
                foreach (var ua in props.UnicastAddresses)
                {
                    IPAddress a = ua.Address;
                    if (a.AddressFamily != AddressFamily.InterNetwork) continue;
                    if (IPAddress.IsLoopback(a)) continue;
                    byte first = a.GetAddressBytes()[0];
                    if (first == 169) continue;              // link-local 169.254.x.x
                    if (seen.Add(a.ToString())) yield return a;
                }
            }
        }
    }
}
