from common import FileSystemConfig
from topologies.BaseTopology import SimpleTopology

from m5.objects import *
from m5.params import *
import sys
#path to your yaml package
#sys.path.insert(0, "/disk/user/project_dir/gem5/build/Garnet_standalone/python-packages")
import yaml

class AnyNET_XY(SimpleTopology):
    description = "AnyNET_XY"

    def __init__(self, controllers):
        self.nodes = controllers

    def makeTopology(self, options, network, IntLink, ExtLink, Router):
        nodes = self.nodes
        link_latency = options.link_latency
        router_latency = options.router_latency
        num_routers = options.num_cpus

        routers = [Router(router_id=i, latency=router_latency) for i in range(num_routers)]
        network.routers = routers

        link_count = 0
        ext_links = []
        
        # Connect each controller to a router (2:1 mapping assumed)
        for i, node in enumerate(nodes):
            router_id = i % len(routers)  # wraps around if more nodes than routers
            ext_links.append(
                ExtLink(
                    link_id=link_count,
                    ext_node=node,
                int_node=routers[router_id],
                latency=link_latency,
                )
            )
            link_count += 1

        # Read allowed inter-router connections from YAML
        allowed_set = set()
        with open(options.config_file, "r") as f:
            allowed_links = yaml.safe_load(f).get("connections", {})
            for chiplet, connect_list in allowed_links.items():
                for conn in connect_list:
                    allowed_set.add(tuple(conn))

        int_links = []

        def append_link(src, dst, outport, inport, weight):
            nonlocal link_count
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[src],
                    dst_node=routers[dst],
                    src_outport=outport,
                    dst_inport=inport,
                    latency=link_latency,
                    weight=weight,
                )
            )
            link_count += 1

        # Create internal links based on YAML
        for (src, dst) in allowed_set:
            append_link(src, dst, "Link", "Link", 1)

        network.ext_links = ext_links
        network.int_links = int_links

    def registerTopology(self, options):
        for i in range(options.num_cpus):
            FileSystemConfig.register_node(
                [i], MemorySize(options.mem_size) // options.num_cpus, i
            )
