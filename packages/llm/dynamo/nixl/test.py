#!/usr/bin/env python3
print('testing nixl...')
import os

import numpy as np
import torch

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

# Configure logging
logger = get_logger(__name__)


if __name__ == "__main__":
    buf_size = 256
    # Allocate memory and register with NIXL

    logger.info("Using NIXL Plugins from:\n%s", os.environ["NIXL_PLUGIN_DIR"])

    # Example using nixl_agent_config
    agent_config = nixl_agent_config(backends=["UCX"])
    nixl_agent1 = nixl_agent("target", agent_config)

    plugin_list = nixl_agent1.get_plugin_list()
    assert "UCX" in plugin_list

    logger.info(
        "Plugin parameters:\n%s\n%s",
        nixl_agent1.get_plugin_mem_types("UCX"),
        nixl_agent1.get_plugin_params("UCX"),
    )

    logger.info(
        "Backend parameters:\n%s\n%s",
        nixl_agent1.get_backend_mem_types("UCX"),
        nixl_agent1.get_backend_params("UCX"),
    )

    addr1 = nixl_utils.malloc_passthru(buf_size * 2)
    addr2 = addr1 + buf_size

    agent1_addrs = [(addr1, buf_size, 0), (addr2, buf_size, 0)]
    agent1_strings = [(addr1, buf_size, 0, "a"), (addr2, buf_size, 0, "b")]

    agent1_reg_descs = nixl_agent1.get_reg_descs(agent1_strings, "DRAM")
    agent1_xfer_descs = nixl_agent1.get_xfer_descs(agent1_addrs, "DRAM")

    # Prefer numpy arrays for performance
    agent1_addrs_np = np.array(agent1_addrs)
    agent1_xfer_descs_np = nixl_agent1.get_xfer_descs(agent1_addrs_np, "DRAM")
    agent1_reg_descs_np = nixl_agent1.get_reg_descs(agent1_addrs_np, "DRAM")

    assert agent1_xfer_descs == agent1_xfer_descs_np
    assert agent1_reg_descs == agent1_reg_descs_np
    logger.debug(
        "Registration descriptors: %s %s", agent1_reg_descs, agent1_reg_descs_np
    )

    # Just for tensor test
    tensors = [torch.zeros(10, dtype=torch.float32) for _ in range(2)]
    agent1_tensor_reg_descs = nixl_agent1.get_reg_descs(tensors)
    agent1_tensor_xfer_descs = nixl_agent1.get_xfer_descs(tensors)

    assert nixl_agent1.register_memory(agent1_reg_descs) is not None

    # Example using default configs, which is UCX backend only
    nixl_agent2 = nixl_agent("initiator", None)
    addr3 = nixl_utils.malloc_passthru(buf_size * 2)
    addr4 = addr3 + buf_size

    agent2_addrs = [(addr3, buf_size, 0), (addr4, buf_size, 0)]
    agent2_strings = [(addr3, buf_size, 0, "a"), (addr4, buf_size, 0, "b")]

    agent2_reg_descs = nixl_agent2.get_reg_descs(agent2_strings, "DRAM")
    agent2_xfer_descs = nixl_agent2.get_xfer_descs(agent2_addrs, "DRAM")

    agent2_descs = nixl_agent2.register_memory(agent2_reg_descs)
    assert agent2_descs is not None

    # Exchange metadata
    meta = nixl_agent1.get_agent_metadata()
    remote_name = nixl_agent2.add_remote_agent(meta)
    logger.info("Loaded name from metadata: %s", remote_name)

    serdes = nixl_agent1.get_serialized_descs(agent1_reg_descs)
    src_descs_recvd = nixl_agent2.deserialize_descs(serdes)
    assert src_descs_recvd == agent1_reg_descs

    # initialize transfer mode
    xfer_handle_1 = nixl_agent2.initialize_xfer(
        "READ", agent2_xfer_descs, agent1_xfer_descs, remote_name, b"UUID1"
    )
    if not xfer_handle_1:
        logger.error("Creating transfer failed.")
        exit()

    # test multiple postings
    for _ in range(2):
        state = nixl_agent2.transfer(xfer_handle_1)
        assert state != "ERR"

        target_done = False
        init_done = False

        while (not init_done) or (not target_done):
            if not init_done:
                state = nixl_agent2.check_xfer_state(xfer_handle_1)
                if state == "ERR":
                    logger.error("Transfer got to Error state.")
                    exit()
                elif state == "DONE":
                    init_done = True
                    logger.info("Initiator done")

            if not target_done:
                if nixl_agent1.check_remote_xfer_done("initiator", b"UUID1"):
                    target_done = True
                    logger.info("Target done")

    # prep transfer mode
    local_prep_handle = nixl_agent2.prep_xfer_dlist(
        "NIXL_INIT_AGENT", [(addr3, buf_size, 0), (addr4, buf_size, 0)], "DRAM"
    )
    remote_prep_handle = nixl_agent2.prep_xfer_dlist(
        remote_name, agent1_xfer_descs, "DRAM"
    )

    assert local_prep_handle != 0
    assert remote_prep_handle != 0

    # test send_notif

    test_notif = str.encode("DESCS: ") + serdes
    nixl_agent2.send_notif(remote_name, test_notif)

    logger.info("sent notif: \n%s", test_notif)

    notif_recv = False

    while not notif_recv:
        notif_map = nixl_agent1.get_new_notifs()
        if "initiator" in notif_map:
            logger.info("received message from initiator")
            for msg in notif_map["initiator"]:
                if msg == test_notif:
                    notif_recv = True

    logger.info("notif test complete, doing transfer 2")

    xfer_handle_2 = nixl_agent2.make_prepped_xfer(
        "WRITE", local_prep_handle, [0, 1], remote_prep_handle, [1, 0], b"UUID2"
    )
    if not local_prep_handle or not remote_prep_handle:
        logger.error("Preparing transfer side handles failed.")
        exit()

    if not xfer_handle_2:
        logger.error("Make prepped transfer failed.")
        exit()

    state = nixl_agent2.transfer(xfer_handle_2)
    assert state != "ERR"

    target_done = False
    init_done = False

    logger.info("Transfer 2 started")

    while (not init_done) or (not target_done):
        if not init_done:
            state = nixl_agent2.check_xfer_state(xfer_handle_2)
            if state == "ERR":
                logger.error("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                init_done = True
                logger.info("Initiator done")

        if not target_done:
            if nixl_agent1.check_remote_xfer_done("initiator", b"UUID2"):
                target_done = True
                logger.info("Target done")

    nixl_agent2.release_xfer_handle(xfer_handle_1)
    nixl_agent2.release_xfer_handle(xfer_handle_2)
    nixl_agent2.release_dlist_handle(local_prep_handle)
    nixl_agent2.release_dlist_handle(remote_prep_handle)
    nixl_agent2.remove_remote_agent("target")
    nixl_agent1.deregister_memory(agent1_reg_descs)
    nixl_agent2.deregister_memory(agent2_reg_descs)

    nixl_utils.free_passthru(addr1)
    nixl_utils.free_passthru(addr3)

    logger.info("Test Complete.")
print('nixl OK\n')
